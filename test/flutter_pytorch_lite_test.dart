import 'dart:typed_data';

import 'package:flutter_pytorch_lite/flutter_pytorch_lite.dart';
import 'package:flutter_pytorch_lite/flutter_pytorch_lite_method_channel.dart';
import 'package:flutter_pytorch_lite/flutter_pytorch_lite_platform_interface.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockFlutterPytorchLitePlatform
    with MockPlatformInterfaceMixin
    implements FlutterPytorchLitePlatform {
  @override
  Future<int> load(String filePath) => Future.value(0);

  @override
  Future<void> destroy(int moduleId) => Future.value();

  @override
  Future<IValue> forward(int moduleId, List<IValue> inputs) =>
      Future.value(IValue.from(
          Tensor.fromBlobUint8(Uint8List(5), Int64List.fromList([1, 5]))));
}

void main() {
  final FlutterPytorchLitePlatform initialPlatform =
      FlutterPytorchLitePlatform.instance;

  test('$MethodChannelFlutterPytorchLite is the default instance', () {
    expect(initialPlatform, isInstanceOf<MethodChannelFlutterPytorchLite>());
  });

  test('getPlatformVersion', () async {
    MockFlutterPytorchLitePlatform fakePlatform =
        MockFlutterPytorchLitePlatform();
    FlutterPytorchLitePlatform.instance = fakePlatform;

    await FlutterPytorchLite.load('');
    // expect(await FlutterPytorchLite.load(''), '42');
  });
}
