import 'dart:io';
import 'dart:convert';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';
import 'package:open_filex/open_filex.dart'; // ✅ Correct import

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(MyApp(cameras: cameras));
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  const MyApp({required this.cameras, super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Opti Frontend',
      theme: ThemeData.dark(),
      debugShowCheckedModeBanner: false,
      home: SplashThenCamera(cameras: cameras),
    );
  }
}

class SplashThenCamera extends StatefulWidget {
  final List<CameraDescription> cameras;
  const SplashThenCamera({required this.cameras, super.key});
  @override
  State<SplashThenCamera> createState() => _SplashThenCameraState();
}

class _SplashThenCameraState extends State<SplashThenCamera> {
  bool showSplash = true;

  @override
  void initState() {
    super.initState();
    Future.delayed(const Duration(milliseconds: 800), () {
      setState(() => showSplash = false);
    });
  }

  @override
  Widget build(BuildContext context) {
    return showSplash
        ? const Scaffold(
            body: Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  FlutterLogo(size: 140),
                  SizedBox(height: 20),
                  Text("Opti", style: TextStyle(fontSize: 32)),
                ],
              ),
            ),
          )
        : CameraScreen(cameras: widget.cameras);
  }
}

class CameraScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  const CameraScreen({required this.cameras, super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? controller;
  bool processing = false;
  int cameraIndex = 0;
  Map<String, dynamic>? lastData;
  String? annotatedImgB64;

  @override
  void initState() {
    super.initState();
    initCamera();
  }

  Future<void> initCamera() async {
    await Permission.camera.request();
    controller = CameraController(
      widget.cameras[cameraIndex],
      ResolutionPreset.high,
      enableAudio: false,
    );
    await controller!.initialize();
    setState(() {});
  }

  @override
  void dispose() {
    controller?.dispose();
    super.dispose();
  }

  Future<void> flipCamera() async {
    if (widget.cameras.length < 2) return;
    cameraIndex = (cameraIndex + 1) % widget.cameras.length;
    await controller?.dispose();
    controller = CameraController(
      widget.cameras[cameraIndex],
      ResolutionPreset.high,
      enableAudio: false,
    );
    await controller!.initialize();
    setState(() {});
  }

  Future<void> captureAndSend() async {
    if (controller == null || !controller!.value.isInitialized || processing) return;
    setState(() => processing = true);

    try {
      final tmp = await getTemporaryDirectory();
      final path = '${tmp.path}/capture_${DateTime.now().millisecondsSinceEpoch}.jpg';
      final XFile raw = await controller!.takePicture();
      await raw.saveTo(path);

      final bytes = await File(path).readAsBytes();
      final b64 = base64Encode(bytes);

      // 👉 Replace with your backend IP
      final uri = Uri.parse("http://192.168.1.38:5000/process");

      final resp = await http.post(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'image_b64': b64}),
      );

      if (resp.statusCode == 200) {
        final data = jsonDecode(resp.body);
        if (data['error'] != null) {
          ScaffoldMessenger.of(context)
              .showSnackBar(SnackBar(content: Text('Error: ${data['error']}')));
        } else {
          setState(() {
            lastData = Map<String, dynamic>.from(data);
            annotatedImgB64 = data['annotated_image_b64'];
          });
        }
      } else {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Server error ${resp.statusCode}')));
      }
    } catch (e) {
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('Error: $e')));
    } finally {
      setState(() => processing = false);
    }
  }

  // 📄 DOWNLOAD PDF FUNCTION
  Future<void> downloadPDF() async {
    if (lastData == null) {
      ScaffoldMessenger.of(context)
          .showSnackBar(const SnackBar(content: Text("⚠️ Please capture first.")));
      return;
    }

    try {
      final url = Uri.parse("http://192.168.1.38:5000/download_pdf"); // 👈 Your backend IP
      final response = await http.post(
        url,
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "optician_name": "OptiMate",
          "customer_name": "John Doe",
          "order_no": "ORD-${DateTime.now().millisecondsSinceEpoch}",
          "material": "1.56",
          "coating": "Blue Cut",
          "type": "Single Vision",
          "dia": "65",
          "sph_re": "-1.25",
          "cyl_re": "-0.50",
          "axis_re": "180",
          "add_re": "0",
          "sph_le": "-1.00",
          "cyl_le": "-0.25",
          "axis_le": "170",
          "add_le": "0",
          "A_mm": lastData?["A_mm"] ?? 0,
          "B_mm": lastData?["B_mm"] ?? 0,
          "DBL_mm": lastData?["DBL_mm"] ?? 0,
          "pd_mm": lastData?["pd_mm"] ?? 0,
          "additional_info": "UV protected lenses",
        }),
      );

      if (response.statusCode == 200) {
        final bytes = response.bodyBytes;
        final dir = await getApplicationDocumentsDirectory();
        final file = File("${dir.path}/order.pdf");
        await file.writeAsBytes(bytes);

        print("✅ PDF saved: ${file.path}");
        final result = await OpenFilex.open(file.path); // 👈 Automatically opens PDF
        if (result.type != ResultType.done) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text("⚠️ Could not open PDF: ${result.message}")),
          );
        }
      } else {
        print("❌ PDF generation failed: ${response.statusCode}");
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Server Error: ${response.statusCode}")),
        );
      }
    } catch (e) {
      print("⚠️ Error while downloading PDF: $e");
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text("Error: $e")));
    }
  }

  @override
  Widget build(BuildContext context) {
    if (controller == null || !controller!.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      body: Stack(
        children: [
          CameraPreview(controller!),

          // Blur outside oval
          Positioned.fill(
            child: ClipPath(
              clipper: OvalHoleClipper(),
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                child: Container(color: Colors.black.withOpacity(0.4)),
              ),
            ),
          ),

          // Buttons
          Positioned(
            bottom: 30,
            left: 20,
            right: 20,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                FloatingActionButton(
                  heroTag: "flip",
                  onPressed: flipCamera,
                  child: const Icon(Icons.flip_camera_ios),
                ),
                FloatingActionButton.large(
                  heroTag: "capture",
                  onPressed: captureAndSend,
                  child: processing
                      ? const CircularProgressIndicator(color: Colors.white)
                      : const Icon(Icons.camera_alt),
                ),
                const SizedBox(width: 56),
              ],
            ),
          ),

          if (lastData != null)
            Positioned(
              top: 40,
              left: 16,
              right: 16,
              child: Card(
                color: Colors.black54,
                child: Padding(
                  padding: const EdgeInsets.all(10),
                  child: Column(
                    children: [
                      Text(
                        lastData!.entries
                            .where((e) => e.key != "annotated_image_b64")
                            .map((e) => "${e.key}: ${e.value}")
                            .join("\n"),
                        style: const TextStyle(fontSize: 14),
                      ),
                      const SizedBox(height: 10),
                      ElevatedButton.icon(
                        onPressed: downloadPDF,
                        icon: const Icon(Icons.picture_as_pdf),
                        label: const Text("Download PDF"),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.deepPurple,
                          foregroundColor: Colors.white,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class OvalHoleClipper extends CustomClipper<Path> {
  @override
  Path getClip(Size size) {
    final path = Path()..addRect(Rect.fromLTWH(0, 0, size.width, size.height));
    final holeRect = Rect.fromCenter(
      center: size.center(Offset.zero),
      width: size.width * 0.72,
      height: size.height * 0.52,
    );
    final hole = Path()..addOval(holeRect);
    return Path.combine(PathOperation.difference, path, hole);
  }

  @override
  bool shouldReclip(covariant CustomClipper<Path> oldClipper) => false;
}
