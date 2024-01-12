import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_pytorch_lite/flutter_pytorch_lite.dart';
import 'package:flutter_pytorch_lite/flutter_pytorch_lite_method_channel.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  MethodChannelFlutterPytorchLite platform = MethodChannelFlutterPytorchLite();
  const MethodChannel channel = MethodChannel('flutter_pytorch_lite');

  setUp(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
      channel,
      (MethodCall methodCall) async {
        if (methodCall.method == 'load') {
          return null;
        } else if (methodCall.method == 'destroy') {
          return null;
        } else if (methodCall.method == 'forward') {
          return Tensor.fromBlobFloat32(
              Float32List(1 * 8), Int64List.fromList([1, 8]));
        }
        return null;
      },
    );
  });

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, null);
  });

  test('load', () async {
    await platform.load('');
    expect(
        await platform.forward(Tensor.fromBlobFloat32(
            Float32List(1 * 3 * 224 * 224),
            Int64List.fromList([1, 3, 224, 224]))),
        Tensor.fromBlobFloat32(Float32List(1 * 8), Int64List.fromList([1, 8])));
  });
}
