import 'dart:typed_data';
import 'flutter_pytorch_lite_platform_interface.dart';

part 'core.dart';

/// PyTorch Lite
///
/// @author guoweifeng
/// @date 2024/1/10 10:37
class FlutterPytorchLite {
  static Future<void> load(String filePath) {
    return FlutterPytorchLitePlatform.instance.load(filePath);
  }

  static Future<void> destroy() {
    return FlutterPytorchLitePlatform.instance.destroy();
  }

  static Future<Tensor> forward(Tensor tensor) {
    return FlutterPytorchLitePlatform.instance.forward(tensor);
  }
}
