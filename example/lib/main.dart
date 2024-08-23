import 'dart:async';
import 'dart:io';
import 'dart:math';
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
    helper.initHelper().then((_) {
      classified();
    });
  }

  Future<void> classified() async {
    ui.Image image = await TensorImageUtils.imageProviderToImage(assetImage);
    classification = await helper.inferenceImage(image);

    if (!mounted) return;
    setState(() {});
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
                        .map((key) =>
                            '$key: ${classification?[key]?.toStringAsFixed(2)}')
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

  final Int64List inputShape = Int64List.fromList([1, 3, 224, 224]);
  final Int64List outputShape = Int64List.fromList([1, 1000]);
  late final List<String> labels;
  Module? mModule;

  // Load model
  Future<void> _loadModel() async {
    final filePath = '${Directory.systemTemp.path}/model.ptl';
    File(filePath).writeAsBytesSync(await _getBuffer(modelPath));
    mModule = await FlutterPytorchLite.load(filePath);
    // mModule = await FlutterPytorchLite.load('notExistPath.ptl');

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
    await _loadLabels();
    await _loadModel();
  }

  // inference still image
  Future<Map<String, double>> inferenceImage(ui.Image image) async {
    // input tensor
    Tensor inputTensor = await TensorImageUtils.imageToFloat32Tensor(
      image,
      width: inputShape[3],
      height: inputShape[2],
    );

    // Forward
    IValue input = IValue.from(inputTensor);
    IValue output = await mModule!.forward([input]);

    // output tensor
    Tensor outputTensor = output.toTensor();

    // Get output tensor
    final result = outputTensor.dataAsFloat32List;

    // probabilities
    final prob = softmax(result);

    // Set classification map {label: points}
    var classification = <String, double>{};
    for (var i = 0; i < prob.length; i++) {
      if (prob[i] != 0) {
        // Set label: points
        classification[labels[i]] = prob[i];
      }
    }

    // top 5 indices
    final top5i = (classification.entries.toList()
          ..sort((a, b) => b.value.compareTo(a.value)))
        .getRange(0, 5)
        // .map((e) => MapEntry(e.key, (e.value * 100).toInt() / 100))
        .toList();
    return Map.fromEntries(top5i);
  }

  List<double> softmax(List<double> logits) {
    // Step 1: Compute the exponential of each element
    List<double> expValues = logits.map((x) => exp(x)).toList();

    // Step 2: Compute the sum of all exponentials
    double sumExpValues = expValues.reduce((a, b) => a + b);

    // Step 3: Normalize each value by the sum of exponentials
    List<double> probabilities =
        expValues.map((x) => x / sumExpValues).toList();

    return probabilities;
  }

  Future<void> close() async {
    await mModule?.destroy();
  }
}
