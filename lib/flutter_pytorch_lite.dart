import 'dart:typed_data';

import 'flutter_pytorch_lite_platform_interface.dart';

export 'utils.dart';

part 'core.dart';

/// PyTorch Lite
///
/// @author guoweifeng
/// @date 2024/1/10 10:37
class FlutterPytorchLite {
  /// Load model from file
  ///
  /// @param filePath model file path
  static Future<Module> load(String filePath) {
    return FlutterPytorchLitePlatform.instance
        .load(filePath)
        .then((moduleId) => Module._(moduleId));
  }
}

/// Module
class Module {
  final int moduleId;

  const Module._(this.moduleId);

  /// Forward
  ///
  /// @param input input
  /// @return output
  Future<IValue> forward(List<IValue> inputs) {
    return FlutterPytorchLitePlatform.instance.forward(moduleId, inputs);
  }

  /// Destroy model
  Future<void> destroy() {
    return FlutterPytorchLitePlatform.instance.destroy(moduleId);
  }
}
