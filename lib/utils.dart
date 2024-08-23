import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import 'flutter_pytorch_lite.dart';

class TensorImageUtils {
  TensorImageUtils._();

  /// Convert an [Image] to a [Tensor]
  static Future<Tensor> imageToFloat32Tensor(
    final ui.Image image, {
    int x = 0,
    int y = 0,
    int? width,
    int? height,
    final MemoryFormat memoryFormat = MemoryFormat.contiguous,
  }) async {
    if (memoryFormat != MemoryFormat.contiguous &&
        memoryFormat != MemoryFormat.channelsLast) {
      throw ArgumentError('Unsupported memory format: $memoryFormat');
    }
    if (width == null || height == null) {
      width = image.width;
      height = image.height;
    }
    final bitmap = resizeImage(image, width, height);
    final int pixelsCount = height * width;
    const int channels = 3;

    // rgba
    const pixelFormat = ui.ImageByteFormat.rawExtendedRgba128;
    final pixels =
        (await bitmap.toByteData(format: pixelFormat))!.buffer.asFloat32List();

    // MemoryFormat
    //   - contiguous:
    //       - inputShape: 1*channels*height*width
    //       - [r0, r1, ..., g0, g1, ..., b0, b1, ...]
    //   - channelsLast:
    //       - inputShape: 1*channels*height*width
    //       - [r0, g0, b0, r1, g1, b1, ...]
    final Int64List inputShape;
    final double Function(int) generator;
    if (memoryFormat == MemoryFormat.contiguous) {
      inputShape = Int64List.fromList([1, channels, height, width]);
      generator = (index) {
        final pixelIdx = index % pixelsCount;
        final rgbIdx = index ~/ pixelsCount;
        return pixels[pixelIdx * 4 + rgbIdx];
        // final x = index % width!;
        // final y = (index ~/ width) % height!;
        // final channel = index ~/ (height * width);
        // return pixels[(y * width + x) * 4 + channel];
      };
    } else {
      inputShape = Int64List.fromList([1, height, width, channels]);
      generator = (index) {
        final pixelIdx = index ~/ channels;
        final rgbIdx = index % channels;
        return pixels[pixelIdx * 4 + rgbIdx];
        // final positionInHeightWidth = index ~/ channels;
        // final x = positionInHeightWidth ~/ height!;
        // final y = positionInHeightWidth % height;
        // final channel = index % channels;
        // return pixels[(y * width! + x) * 4 + channel];
      };
    }

    // rgb
    final imageMatrix =
        Float32List.fromList(List.generate(pixelsCount * channels, generator));

    // tensor
    return Tensor.fromBlobFloat32(imageMatrix, inputShape,
        memoryFormat: memoryFormat);
  }

  /// Resizes an [ui.Image] to a given [targetWidth] and [targetHeight]
  static ui.Image resizeImage(
    ui.Image image,
    int targetWidth,
    int targetHeight, {
    int x = 0,
    int y = 0,
  }) {
    if (image.width == targetWidth && image.height == targetHeight) {
      return image;
    }
    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(recorder);

    canvas.drawImageRect(
      image,
      ui.Rect.fromLTRB(x.toDouble(), y.toDouble(), image.width.toDouble(),
          image.height.toDouble()),
      ui.Rect.fromLTRB(0, 0, targetWidth.toDouble(), targetHeight.toDouble()),
      ui.Paint(),
    );

    return recorder.endRecording().toImageSync(targetWidth, targetHeight);
  }

  /// Convert [ImageProvider] to [ui.Image]
  /// e.g. TensorImageUtils.imageProviderToImage(const AssetImage('assets/images/image.png'))
  static Future<ui.Image> imageProviderToImage(ImageProvider image) {
    Completer<ui.Image> completer = Completer.sync();
    image.resolve(ImageConfiguration.empty).addListener(
        ImageStreamListener((ImageInfo image, bool synchronousCall) {
      if (!completer.isCompleted) completer.complete(image.image);
    }));
    return completer.future;
  }
}
