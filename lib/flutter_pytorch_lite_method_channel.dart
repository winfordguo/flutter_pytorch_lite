import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'flutter_pytorch_lite.dart';
import 'flutter_pytorch_lite_platform_interface.dart';

/// PyTorch Lite
/// An implementation of [FlutterPytorchLitePlatform] that uses method channels.
///
/// @author guoweifeng
/// @date 2024/1/10 10:37
class MethodChannelFlutterPytorchLite extends FlutterPytorchLitePlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('flutter_pytorch_lite');

  @override
  Future<void> load(String filePath) async {
    return methodChannel.invokeMethod<void>('load', {
      'filePath': filePath,
    });
  }

  @override
  Future<void> destroy() {
    return methodChannel.invokeMethod<void>('destroy');
  }

  @override
  Future<Tensor> forward(Tensor tensor) async {
    final map = await methodChannel.invokeMethod('forward', tensor.toMap());
    return Tensor.fromMap(map);
  }
}
