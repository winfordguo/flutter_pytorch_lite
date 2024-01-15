import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_pytorch_lite/flutter_pytorch_lite.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  static const assetImage = AssetImage('assets/images/image.png');

  ImageClassificationHelper helper = ImageClassificationHelper();
  Map<String, double>? classification;

  @override
  void initState() {
    super.initState();
    helper.initHelper();

    classified();
  }

  Future<void> classified() async {
    ui.Image image = await _loadImage();
    classification = await helper.inferenceImage(image);

    if (!mounted) return;
    setState(() {});
  }

  Future<ui.Image> _loadImage() {
    Completer<ui.Image> completer = Completer.sync();
    assetImage.resolve(ImageConfiguration.empty).addListener(
        ImageStreamListener((ImageInfo image, bool synchronousCall) {
      if (!completer.isCompleted) completer.complete(image.image);
    }));
    return completer.future;
  }

  @override
  void dispose() {
    super.dispose();
    helper.close();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('Flutter PyTorch Lite'),
        ),
        body: Center(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const Text('A simple image classification application!\n'),
              const Image(image: assetImage),
              if (classification != null)
                Expanded(
                    child: SingleChildScrollView(
                  child: Text(
                    classification!.keys
                        .map((key) => '$key: ${classification?[key]}')
                        .join('\n'),
                    textAlign: TextAlign.center,
                  ),
                )),
            ],
          ),
        ),
      ),
    );
  }
}

class ImageClassificationHelper {
  static const modelPath = 'assets/models/model.ptl';
  static const labelsPath = 'assets/models/words.txt';

  late final List<String> labels;
  final Int64List inputShape = Int64List.fromList([1, 3, 224, 224]);
  final Int64List outputShape = Int64List.fromList([1, 1000]);

  // Load model
  Future<void> _loadModel() async {
    final filePath = '${Directory.systemTemp.path}/model.ptl';
    File(filePath).writeAsBytesSync(await _getBuffer(modelPath));
    await FlutterPytorchLite.load(filePath);

    print('Interpreter loaded successfully');
  }

  /// Get byte buffer
  static Future<Uint8List> _getBuffer(String assetFileName) async {
    ByteData rawAssetFile = await rootBundle.load(assetFileName);
    final rawBytes = rawAssetFile.buffer.asUint8List();
    return rawBytes;
  }

  // Load labels from assets
  Future<void> _loadLabels() async {
    final labelTxt = await rootBundle.loadString(labelsPath);
    labels = labelTxt.split('\n');
  }

  Future<void> initHelper() async {
    _loadLabels();
    _loadModel();
  }

  // inference still image
  Future<Map<String, double>> inferenceImage(ui.Image image) async {
    final height = inputShape[2];
    final width = inputShape[3];
    ui.Image imageInput = await _resizeImage(image, width, height);

    // rgba
    final pixels = (await imageInput.toByteData(
            format: ui.ImageByteFormat.rawExtendedRgba128))!
        .buffer
        .asFloat32List();
    // rgb
    final imageMatrix = Float32List.fromList(
        List.generate(inputShape[0] * inputShape[1] * height * width, (index) {
      final pixelIdx = index ~/ inputShape[1];
      final rgbIdx = index % inputShape[1];
      return pixels[pixelIdx * 4 + rgbIdx];
    }));
    Tensor inputTensor = Tensor.fromBlobFloat32(imageMatrix, inputShape);

    // Forward
    Tensor outputTensor = await FlutterPytorchLite.forward(inputTensor);

    // Get output tensor
    final result = outputTensor.dataAsFloat32List;

    // Set classification map {label: points}
    var classification = <String, double>{};
    for (var i = 0; i < result.length; i++) {
      if (result[i] != 0) {
        // Set label: points
        classification[labels[i]] = result[i];
      }
    }
    return classification;
  }

  // Resizes an [ui.Image] to a given [targetWidth] and [targetHeight]
  Future<ui.Image> _resizeImage(
      ui.Image image, int targetWidth, int targetHeight) {
    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(recorder);

    canvas.drawImageRect(
      image,
      ui.Rect.fromLTRB(0, 0, image.width.toDouble(), image.height.toDouble()),
      ui.Rect.fromLTRB(0, 0, targetWidth.toDouble(), targetHeight.toDouble()),
      ui.Paint(),
    );

    return recorder.endRecording().toImage(targetWidth, targetHeight);
  }

  Future<void> close() async {
    await FlutterPytorchLite.destroy();
  }
}
