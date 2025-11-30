import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:typed_data';
import 'result_screen.dart';

class CameraScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const CameraScreen({super.key, required this.cameras});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? controller;
  int selectedCamera = 0;
  bool busy = false;

  @override
  void initState() {
    super.initState();
    if (widget.cameras.isNotEmpty) {
      initCamera(widget.cameras[selectedCamera]);
    }
  }

  Future<void> initCamera(CameraDescription cam) async {
    controller = CameraController(
      cam,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    await controller!.initialize();
    if (!mounted) return;

    setState(() {});
  }

  void toggleCamera() {
    if (widget.cameras.length < 2) return;
    selectedCamera = (selectedCamera + 1) % widget.cameras.length;
    initCamera(widget.cameras[selectedCamera]);
  }

  Future<void> captureAndSend() async {
    if (controller == null || !controller!.value.isInitialized) return;

    try {
      setState(() => busy = true);

      XFile file = await controller!.takePicture();
      Uint8List bytes = await file.readAsBytes();
      String b64 = base64Encode(bytes);

      var res = await http.post(
        Uri.parse("http://10.158.255.79:5000/process"),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"image_b64": b64}),
      );

      if (res.statusCode == 200 && mounted) {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => ResultScreen(
              resultData: Map<String, dynamic>.from(jsonDecode(res.body)),
            ),
          ),
        );
      }
    } catch (e) {
      print("ERROR: $e");
    } finally {
      if (mounted) setState(() => busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (controller == null || !controller!.value.isInitialized) {
      return Scaffold(
        body: Center(
          child: CircularProgressIndicator(
            color: Colors.yellow.shade700,
          ),
        ),
      );
    }

    bool isFrontCamera =
        controller!.description.lensDirection == CameraLensDirection.front;

    return Scaffold(
      body: Stack(
        children: [
          Center(child: CameraPreview(controller!)),

          Positioned.fill(
            child: CustomPaint(painter: OvalOverlayPainter()),
          ),

          if (isFrontCamera)
            Positioned(
              top: 40,
              left: 0,
              right: 0,
              child: Center(
                child: Container(
                  width: 14,
                  height: 14,
                  decoration: const BoxDecoration(
                    color: Colors.red,
                    shape: BoxShape.circle,
                  ),
                ),
              ),
            ),

          Positioned(
            top: 70,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 18, vertical: 8),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.35),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Text(
                  "Align your face inside the oval",
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 17,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ),
          ),

          Positioned(
            top: 40,
            right: 20,
            child: FloatingActionButton(
              backgroundColor: Colors.black.withOpacity(0.4),
              mini: true,
              onPressed: toggleCamera,
              child: const Icon(Icons.cameraswitch, color: Colors.white),
            ),
          ),

          Positioned(
            bottom: 40,
            left: 0,
            right: 0,
            child: Center(
              child: GestureDetector(
                onTap: busy ? null : captureAndSend,
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  width: busy ? 70 : 85,
                  height: busy ? 70 : 85,
                  decoration: BoxDecoration(
                    color: Colors.yellow.shade600,
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(
                        color: Colors.yellow.shade800.withOpacity(0.4),
                        blurRadius: 20,
                        spreadRadius: 2,
                      ),
                    ],
                  ),
                  child: Center(
                    child: busy
                        ? const CircularProgressIndicator(
                            color: Colors.black, strokeWidth: 3)
                        : const Icon(Icons.camera_alt,
                            size: 35, color: Colors.black),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class OvalOverlayPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final overlay = Paint()..color = Colors.black.withOpacity(0.90);
    final path =
        Path()..addRect(Rect.fromLTWH(0, 0, size.width, size.height));

    final oval = Rect.fromCenter(
      center: size.center(Offset.zero),
      width: size.width * 0.78,
      height: size.height * 0.58,
    );

    path.addOval(oval);
    path.fillType = PathFillType.evenOdd;

    canvas.drawPath(path, overlay);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
