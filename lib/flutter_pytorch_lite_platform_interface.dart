import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'flutter_pytorch_lite.dart';
import 'flutter_pytorch_lite_method_channel.dart';

/// PyTorch Lite
///
/// @author guoweifeng
/// @date 2024/1/10 10:37
abstract class FlutterPytorchLitePlatform extends PlatformInterface {
  /// Constructs a FlutterPytorchLitePlatform.
  FlutterPytorchLitePlatform() : super(token: _token);

  static final Object _token = Object();

  static FlutterPytorchLitePlatform _instance =
      MethodChannelFlutterPytorchLite();

  /// The default instance of [FlutterPytorchLitePlatform] to use.
  ///
  /// Defaults to [MethodChannelFlutterPytorchLite].
  static FlutterPytorchLitePlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [FlutterPytorchLitePlatform] when
  /// they register themselves.
  static set instance(FlutterPytorchLitePlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<int> load(String filePath) {
    throw UnimplementedError('load() has not been implemented.');
  }

  Future<void> destroy(int moduleId) {
    throw UnimplementedError('destroy() has not been implemented.');
  }

  Future<IValue> forward(int moduleId, List<IValue> inputs) {
    throw UnimplementedError('forward() has not been implemented.');
  }
}
