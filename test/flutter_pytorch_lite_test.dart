import 'package:flutter_pytorch_lite/flutter_pytorch_lite.dart';
import 'package:flutter_pytorch_lite/flutter_pytorch_lite_method_channel.dart';
import 'package:flutter_pytorch_lite/flutter_pytorch_lite_platform_interface.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockFlutterPytorchLitePlatform
    with MockPlatformInterfaceMixin
    implements FlutterPytorchLitePlatform {
  @override
  Future<void> load(String filePath) => Future.value();

  @override
  Future<void> destroy() => Future.value();

  @override
  Future<Tensor> forward(Tensor tensor) => Future.value();
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
