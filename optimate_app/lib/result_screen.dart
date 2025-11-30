// result_screen.dart
import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'package:flutter/rendering.dart';


// Make sure this import points to your details screen file.
// If your details screen is in another file, change the path accordingly.


class ResultScreen extends StatefulWidget {
  final Map<String, dynamic>? resultData;

  const ResultScreen({super.key, required this.resultData});

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  ui.Image? _cachedImage;
  Size? _imageSize;

  Offset? leftMarkerOffset;
  Offset? rightMarkerOffset;

  Offset? _originalLeftMarker;
  Offset? _originalRightMarker;

  double originalLeftPD = 0.0;
  double originalRightPD = 0.0;

  bool editMode = false;

  // Key used for exporting the rendered image (image + markers)
  final GlobalKey _repaintKey = GlobalKey();

  @override
  void initState() {
    super.initState();
    final data = widget.resultData ?? {};
    originalLeftPD = (data['pd_left_mm'] ?? 0).toDouble();
    originalRightPD = (data['pd_right_mm'] ?? 0).toDouble();
    _loadImage();
  }

  Future<void> _loadImage() async {
    final bytes = widget.resultData?['image_b64'];
    if (bytes == null) return;

    try {
      final Uint8List imageBytes = base64Decode(bytes);
      final ui.Image img = await _decodeUiImage(imageBytes);
      if (!mounted) return;

      setState(() {
        _cachedImage = img;
        _imageSize = Size(img.width.toDouble(), img.height.toDouble());

        // initialize marker positions relative to screen width and image height
        final w = MediaQuery.of(context).size.width;
        final h = _imageSize!.height * (w / _imageSize!.width);

        leftMarkerOffset ??= Offset(w * 0.3, h / 2);
        rightMarkerOffset ??= Offset(w * 0.7, h / 2);

        _originalLeftMarker ??= leftMarkerOffset;
        _originalRightMarker ??= rightMarkerOffset;
      });
    } catch (e, st) {
      debugPrint('Failed to load image: $e\n$st');
    }
  }

  Future<ui.Image> _decodeUiImage(Uint8List bytes) async {
    final completer = Completer<ui.Image>();
    ui.decodeImageFromList(bytes, (img) => completer.complete(img));
    return completer.future;
  }

  /// Exports the RepaintBoundary as a base64 PNG (image with markers baked in)
  Future<String?> _exportUpdatedImage(GlobalKey boundaryKey) async {
    try {
      if (boundaryKey.currentContext == null) return null;
      final boundary =
          boundaryKey.currentContext!.findRenderObject() as RenderRepaintBoundary?;

      if (boundary == null) return null;

      // If the boundary still needs paint, wait a tick
      if (boundary.debugNeedsPaint) {
        await Future.delayed(const Duration(milliseconds: 30));
      }

      final pixelRatio = MediaQuery.of(context).devicePixelRatio;
      final ui.Image img = await boundary.toImage(pixelRatio: pixelRatio);
      final ByteData? byteData =
          await img.toByteData(format: ui.ImageByteFormat.png);
      if (byteData == null) return null;

      return base64Encode(byteData.buffer.asUint8List());
    } catch (e) {
      debugPrint("EXPORT ERROR: $e");
      return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    final data = widget.resultData ?? {};
    final A_mm = (data["A_mm"] ?? 0).toDouble();
    final B_mm = (data["B_mm"] ?? 0).toDouble();
    final DBL_mm = (data["DBL_mm"] ?? 0).toDouble();

    double pxToMm = 1.0;
    if (_imageSize != null) {
      final w = MediaQuery.of(context).size.width;
      // Keep your original scale assumption (adjust if needed)
      pxToMm = 25.0 / w;
    }

    double pdLeft = originalLeftPD;
    double pdRight = originalRightPD;

    if (leftMarkerOffset != null && _originalLeftMarker != null) {
      pdLeft += (leftMarkerOffset!.dx - _originalLeftMarker!.dx) * pxToMm;
    }
    if (rightMarkerOffset != null && _originalRightMarker != null) {
      pdRight += (rightMarkerOffset!.dx - _originalRightMarker!.dx) * pxToMm;
    }

    final pdTotal = pdLeft + pdRight;

    String mm2(double v) => "${v.toStringAsFixed(1)} mm";

    return Scaffold(
      resizeToAvoidBottomInset: false,
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios, color: Colors.white),
          onPressed: () => Navigator.of(context).maybePop(),
        ),
        title: const Text("Face Region (Adjust)",
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600)),
        actions: [
          IconButton(
            icon: Icon(editMode ? Icons.check : Icons.edit, color: Colors.white),
            onPressed: () => setState(() => editMode = !editMode),
          ),
          IconButton(
            icon: const Icon(Icons.restore, color: Colors.white),
            onPressed: () => setState(() => _restoreMarkers()),
          )
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            _buildImageArea(),
            _buildMetrics(pdTotal, pdLeft, pdRight, A_mm, B_mm, DBL_mm),
          ],
        ),
      ),
    );
  }

  Widget _buildImageArea() {
    if (_cachedImage == null) {
      return Expanded(
        flex: 4,
        child: Center(
          child: CircularProgressIndicator(color: Colors.grey.shade300),
        ),
      );
    }

    return Expanded(
      flex: 4,
      child: LayoutBuilder(builder: (context, c) {
        final w = c.maxWidth;
        final h = _imageSize!.height * (w / _imageSize!.width);

        return Center(
          child: SizedBox(
            width: w,
            height: h,
            child: RepaintBoundary(
              key: _repaintKey,
              child: Stack(
                children: [
                  // Image (fit to box)
                  Positioned.fill(
                    child: FittedBox(
                      fit: BoxFit.contain,
                      alignment: Alignment.topCenter,
                      child: SizedBox(
                        width: _imageSize!.width,
                        height: _imageSize!.height,
                        child: RawImage(image: _cachedImage),
                      ),
                    ),
                  ),

                  // left marker
                  if (leftMarkerOffset != null)
                    _buildMarker(
                      leftMarkerOffset!,
                      (o) => setState(() => leftMarkerOffset = o),
                      editMode,
                      boundary: Rect.fromLTWH(0, 0, w, h),
                    ),

                  // right marker
                  if (rightMarkerOffset != null)
                    _buildMarker(
                      rightMarkerOffset!,
                      (o) => setState(() => rightMarkerOffset = o),
                      editMode,
                      boundary: Rect.fromLTWH(0, 0, w, h),
                    ),
                ],
              ),
            ),
          ),
        );
      }),
    );
  }

  Widget _buildMetrics(double pdTotal, double pdLeft, double pdRight,
      double A, double B, double DBL) {
    String mm(double v) => "${v.toStringAsFixed(1)} mm";

    return Flexible(
      flex: 3,
      child: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
          child: Column(
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.grey.shade900.withOpacity(0.95),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: GridView.count(
                  crossAxisCount: 3,
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  crossAxisSpacing: 10,
                  mainAxisSpacing: 10,
                  childAspectRatio: 1.2,
                  children: [
                    _miniCard("PD", mm(pdTotal)),
                    _miniCard("R-PD", mm(pdRight)),
                    _miniCard("L-PD", mm(pdLeft)),
                    _miniCard("A", mm(A)),
                    _miniCard("B", mm(B)),
                    _miniCard("DBL", mm(DBL)),
                  ],
                ),
              ),
              const SizedBox(height: 14),
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton(
                      onPressed: () {
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(content: Text("Adjustments Applied")),
                        );
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.grey.shade800,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(28),
                        ),
                      ),
                      child: const Text(
                        "Apply Adjustments",
                        style: TextStyle(fontSize: 15),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: ElevatedButton(
                      onPressed: () async {
                        // Export the image (image + markers) and pass it to details screen
                        final updated = await _exportUpdatedImage(_repaintKey);
                        if (updated == null) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(content: Text("Export Failed")),
                          );
                          return;
                        }

                        // NAVIGATE TO RESULTDETAILSSCREEN (expects that file/class exists)
                        Navigator.of(context).push(
                          MaterialPageRoute(
                            builder: (_) => ResultDetailsScreen(
                              imageB64: updated,
                              imageSize: _imageSize,
                              leftMarker: leftMarkerOffset,
                              rightMarker: rightMarkerOffset,
                              computedValues: {
                                'pd_left': pdLeft,
                                'pd_right': pdRight,
                                'pd_total': pdTotal,
                                'A_mm': A,
                                'B_mm': B,
                                'DBL_mm': DBL,
                                'original_left_pd': originalLeftPD,
                                'original_right_pd': originalRightPD,
                              },
                            ),
                          ),
                        );
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.deepPurpleAccent.shade200,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(28),
                        ),
                      ),
                      child: const Text(
                        "Proceed",
                        style: TextStyle(fontSize: 15),
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _restoreMarkers() {
    leftMarkerOffset = _originalLeftMarker;
    rightMarkerOffset = _originalRightMarker;
    setState(() {});
  }

  /// Marker widget with smooth dragging.
  /// onDrag will be called with the new clamped Offset; the caller sets state.
  Widget _buildMarker(
      Offset offset, Function(Offset) onDrag, bool active,
      {Rect? boundary}) {
    Offset clamp(Offset p) {
      if (boundary == null) return p;
      return Offset(
        p.dx.clamp(boundary.left + 10, boundary.right - 10),
        p.dy.clamp(boundary.top + 10, boundary.bottom - 10),
      );
    }

    // For smooth dragging we update the offset from the latest state each onPanUpdate
    return Positioned(
      left: offset.dx - 18,
      top: offset.dy - 18,
      child: GestureDetector(
        onPanUpdate: active
            ? (details) {
                // Add delta to current offset and clamp
                final newPos = clamp(offset + details.delta);
                onDrag(newPos);
              }
            : null,
        child: SizedBox(
          width: 36,
          height: 36,
          child: Center(
            child: Stack(
              alignment: Alignment.center,
              children: [
                Container(width: 20, height: 2, color: Colors.greenAccent),
                Container(width: 2, height: 20, color: Colors.greenAccent),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _miniCard(String label, String value) {
    return Container(
      decoration: BoxDecoration(
        color: const Color.fromARGB(58, 86, 85, 85).withOpacity(0.6),
        borderRadius: BorderRadius.circular(10),
      ),
      padding: const EdgeInsets.all(10),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(label,
              style: TextStyle(
                  fontSize: 12,
                  color: Colors.grey.shade400,
                  fontWeight: FontWeight.w600)),
          const SizedBox(height: 6),
          Text(value,
              style: const TextStyle(
                  fontSize: 14, fontWeight: FontWeight.bold, color: Colors.white)),
        ],
      ),
    );
  }
}


/// Details screen (image preview + computed table + manual fields).
/// CLEANED — NO OVERLAY MARKERS (uses only baked markers)
class ResultDetailsScreen extends StatefulWidget {
  final String? imageB64;
  final Size? imageSize;
  final Offset? leftMarker;
  final Offset? rightMarker;
  final Map<String, dynamic> computedValues;

  const ResultDetailsScreen({
    Key? key,
    required this.imageB64,
    required this.imageSize,
    required this.leftMarker,
    required this.rightMarker,
    required this.computedValues,
  }) : super(key: key);

  @override
  State<ResultDetailsScreen> createState() => _ResultDetailsScreenState();
}

class _ResultDetailsScreenState extends State<ResultDetailsScreen> {
  ui.Image? _img;
  final TextEditingController manualLeftController = TextEditingController();
  final TextEditingController manualRightController = TextEditingController();
  final TextEditingController manualTotalController = TextEditingController();
  final TextEditingController manualAController = TextEditingController();
  final TextEditingController manualBController = TextEditingController();
  final TextEditingController manualDBLController = TextEditingController();

  @override
  void initState() {
    super.initState();
    if (widget.imageB64 != null) _loadImage(widget.imageB64!);
  }

  Future<void> _loadImage(String b64) async {
    try {
      final data = base64Decode(b64);
      final completer = Completer<ui.Image>();
      ui.decodeImageFromList(data, (img) => completer.complete(img));
      final img = await completer.future;
      if (!mounted) return;
      setState(() => _img = img);
    } catch (e) {
      debugPrint("Details image load failed: $e");
    }
  }

  @override
  void dispose() {
    manualLeftController.dispose();
    manualRightController.dispose();
    manualTotalController.dispose();
    manualAController.dispose();
    manualBController.dispose();
    manualDBLController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final computed = widget.computedValues;
    final bg = const Color(0xFF0F0F11);
    final card = Colors.grey.shade900;
    final bright = Colors.white;

    Widget buildRow(String k, dynamic v) {
      final text = (v == null)
          ? '-'
          : (v is num ? '${v.toStringAsFixed(1)} mm' : v.toString());
      return Padding(
        padding: const EdgeInsets.symmetric(vertical: 6),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(k, style: const TextStyle(color: Colors.white70)),
            Text(text,
                style: const TextStyle(
                    color: Colors.white, fontWeight: FontWeight.w600)),
          ],
        ),
      );
    }

    return Scaffold(
      backgroundColor: bg,
      appBar: AppBar(
        backgroundColor: card,
        title: const Text('Result — Image & Readings'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // IMAGE PREVIEW — baked markers only, no overlays
            if (_img == null)
              SizedBox(
                  height: 200,
                  child: Center(child: CircularProgressIndicator()))
            else
              LayoutBuilder(builder: (context, constraints) {
                final w = constraints.maxWidth;
                final h = widget.imageSize != null
                    ? widget.imageSize!.height *
                        (w / widget.imageSize!.width)
                    : _img!.height.toDouble() *
                        (w / _img!.width.toDouble());

                return Container(
                  decoration: BoxDecoration(
                    color: card,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  padding: const EdgeInsets.all(8),
                  child: SizedBox(
                    width: w,
                    height: h,
                    child: RawImage(image: _img),
                  ),
                );
              }),

            const SizedBox(height: 12),

            // COMBINED DATA
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  child: Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: card,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('PD Values',
                            style: TextStyle(
                                fontWeight: FontWeight.bold, color: bright)),
                        const SizedBox(height: 8),
                        buildRow('L-PD', computed['pd_left']),
                        buildRow('R-PD', computed['pd_right']),
                        buildRow('Total PD', computed['pd_total']),
                      ],
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: card,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('A/B/DBL Values',
                            style: TextStyle(
                                fontWeight: FontWeight.bold, color: bright)),
                        const SizedBox(height: 8),
                        buildRow('A', computed['A_mm']),
                        buildRow('B', computed['B_mm']),
                        buildRow('DBL', computed['DBL_mm']),
                      ],
                    ),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 16),

            // MANUAL ENTRY
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                  color: card,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.grey.shade800)),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Manual Entry',
                      style: TextStyle(
                          fontWeight: FontWeight.bold, color: bright)),
                  const SizedBox(height: 8),
                  _field('L-PD (mm)', manualLeftController),
                  const SizedBox(height: 8),
                  _field('R-PD (mm)', manualRightController),
                  const SizedBox(height: 8),
                  _field('Total PD (mm)', manualTotalController),
                  const SizedBox(height: 8),
                  _field('A (mm)', manualAController),
                  const SizedBox(height: 8),
                  _field('B (mm)', manualBController),
                  const SizedBox(height: 8),
                  _field('DBL (mm)', manualDBLController),

                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton(
                          style: ElevatedButton.styleFrom(
                              backgroundColor:
                                  const Color.fromARGB(255, 11, 9, 15)),
                          onPressed: () {
                            ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(
                                    content: Text('Manual values saved')));
                          },
                          child: const Text('Save Manual Values'),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: OutlinedButton(
                          style: OutlinedButton.styleFrom(
                              side: const BorderSide(
                                  color: Color.fromARGB(255, 30, 30, 30))),
                          onPressed: () {
                            manualLeftController.clear();
                            manualRightController.clear();
                            manualTotalController.clear();
                            manualAController.clear();
                            manualBController.clear();
                            manualDBLController.clear();
                            ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(
                                    content: Text('Manual values cleared')));
                          },
                          child: const Text('Clear'),
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 12),
                  ElevatedButton(
                    style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.black),
                    onPressed: () {
                      final manualMap = {
                        'manual_pd_left': manualLeftController.text,
                        'manual_pd_right': manualRightController.text,
                        'manual_pd_total': manualTotalController.text,
                        'manual_A_mm': manualAController.text,
                        'manual_B_mm': manualBController.text,
                        'manual_DBL_mm': manualDBLController.text,
                      };
                      Navigator.of(context).pop(manualMap);
                    },
                    child: const Text('Done — Return Manual Values'),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _field(String label, TextEditingController c) {
    return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: const TextStyle(fontSize: 13, color: Colors.white70)),
          const SizedBox(height: 6),
          TextField(
            controller: c,
            keyboardType:
                const TextInputType.numberWithOptions(decimal: true),
            style: const TextStyle(color: Colors.white),
            decoration: InputDecoration(
              filled: true,
              fillColor: Colors.grey.shade800,
              border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(6),
                  borderSide: BorderSide.none),
              isDense: true,
              contentPadding:
                  const EdgeInsets.symmetric(vertical: 10, horizontal: 8),
            ),
          )
        ]);
  }
}
