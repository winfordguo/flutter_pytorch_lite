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
  Future<int> load(String filePath) async {
    return (await methodChannel.invokeMethod<int>('load', {
      'filePath': filePath,
    }))!;
  }

  @override
  Future<void> destroy(int moduleId) {
    return methodChannel.invokeMethod<void>('destroy', {
      'moduleId': moduleId,
    });
  }

  @override
  Future<IValue> forward(int moduleId, List<IValue> inputs) async {
    final map = await methodChannel.invokeMethod('forward', {
      'moduleId': moduleId,
      'inputs': inputs.map((e) => e.toMap()).toList(),
    });
    return IValue.fromMap(map);
  }
}
