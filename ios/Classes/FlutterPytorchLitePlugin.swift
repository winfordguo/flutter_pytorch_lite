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

    // private Module mModule;
    // private static final ConcurrentHashMap<Integer, Module> mModules = new ConcurrentHashMap<>();
//     private var mModule: TorchModule?
    private var mModules: [Int: TorchModule] = [:]
    
    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "load":
            let arguments = call.arguments as! Dictionary<String, Any>
            let filePath = arguments["filePath"] as! String
            let mModule = TorchModule(fileAtPath: filePath)
            if (mModule == nil) {
                result(FlutterError(code: "loadError",
                                    message: "Pytorch lite load module error",
                                    details: nil))
                return
            }
            mModules[mModule.hashValue] = mModule
            result(mModule.hashValue)
        case "destroy":
            let arguments = call.arguments as! Dictionary<String, Any>
            let moduleId = arguments["moduleId"] as! Int
            let mModule = mModules[moduleId]
            mModules.removeValue(forKey: moduleId)
            mModule?.destroy()

            result(nil)
        case "forward":
            let arguments = call.arguments as! Dictionary<String, Any>
            let moduleId = arguments["moduleId"] as! Int
            let mModule = mModules[moduleId]
            if (mModule == nil) {
                result(FlutterError(code: "forwardError",
                                    message: "Pytorch lite forward module error, module is nil",
                                    details: nil))
                return
            }
            assert(arguments["inputs"] != nil)
            let inputs = arguments["inputs"] as! [Dictionary<String, Any>]
            assert(inputs.count > 0)
            let outputs = mModule?.forward(array: inputs)
            if (outputs == nil) {
                result(FlutterError(code: "forwardError",
                                    message: "Pytorch lite forward module error, outputs is nil",
                                    details: nil))
                return
            }
            result(outputs)
        default:
            result(FlutterMethodNotImplemented)
        }
    }
}
