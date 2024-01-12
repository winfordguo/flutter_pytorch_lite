import Flutter
import UIKit

/**
 * PyTorch Lite
 *
 * @author guoweifeng
 * @date 2024/1/10 10:37
 */
public class FlutterPytorchLitePlugin: NSObject, FlutterPlugin {

    public static func register(with registrar: FlutterPluginRegistrar) {
        // 在后台线程中执行 channel 的 handlers
//        let taskQueue = registrar.messenger.makeBackgroundTaskQueue()
//        let channel = FlutterMethodChannel(name: "flutter_pytorch_lite",
//                                           binaryMessenger: registrar.messenger(),
//                                           codec: FlutterStandardMethodCodec.sharedInstance,
//                                           taskQueue: taskQueue)
        let channel = FlutterMethodChannel(name: "flutter_pytorch_lite", binaryMessenger: registrar.messenger())
        let instance = FlutterPytorchLitePlugin()
        registrar.addMethodCallDelegate(instance, channel: channel)
    }
    
    private var mModule: TorchModule?
    
    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "load":
            let arguments = call.arguments as! Dictionary<String, Any>
            let filePath = arguments["filePath"] as! String
            mModule = TorchModule(fileAtPath: filePath)
            
            result(String(format: "module %d", mModule.hashValue))
        case "destroy":
            mModule?.destroy()
            mModule = nil
            
            result(nil)
        case "forward":
            let input = call.arguments as! Dictionary<String, Any>
            let output = mModule?.forward(map: input)

            result(output)
        default:
            result(FlutterMethodNotImplemented)
        }
    }
}
