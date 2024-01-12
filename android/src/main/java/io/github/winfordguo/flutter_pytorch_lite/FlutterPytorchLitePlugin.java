package io.github.winfordguo.flutter_pytorch_lite;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import org.pytorch.DType;
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.MemoryFormat;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.lang.reflect.Field;
import java.util.HashMap;

import io.flutter.embedding.engine.plugins.FlutterPlugin;
import io.flutter.plugin.common.BinaryMessenger;
import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import io.flutter.plugin.common.MethodChannel.MethodCallHandler;
import io.flutter.plugin.common.MethodChannel.Result;
import io.flutter.plugin.common.StandardMethodCodec;

/**
 * PyTorch Lite
 *
 * @author guoweifeng
 * @date 2024/1/10 10:37
 */
public class FlutterPytorchLitePlugin implements FlutterPlugin, MethodCallHandler {
    /// The MethodChannel that will the communication between Flutter and native Android
    ///
    /// This local reference serves to register the plugin with the Flutter Engine and unregister it
    /// when the Flutter Engine is detached from the Activity
    private MethodChannel channel;

    private Module mModule;

    @Override
    public void onAttachedToEngine(@NonNull FlutterPluginBinding flutterPluginBinding) {
        // 在后台线程中执行 channel 的 handlers
        BinaryMessenger messenger = flutterPluginBinding.getBinaryMessenger();
        BinaryMessenger.TaskQueue taskQueue = messenger.makeBackgroundTaskQueue();
        channel = new MethodChannel(messenger,
                "flutter_pytorch_lite",
                StandardMethodCodec.INSTANCE,
                taskQueue);
        channel.setMethodCallHandler(this);
    }

    @Override
    public void onMethodCall(@NonNull MethodCall call, @NonNull Result result) {
        switch (call.method) {
            case "load":
                String filePath = call.argument("filePath");
                try {
                    mModule = LiteModuleLoader.load(filePath);
                    result.success(mModule.hashCode());
                } catch (Exception e) {
                    result.error("loadError", "Pytorch lite load module error.", e);
                }
                break;
            case "destroy":
                if (mModule != null) {
                    mModule.destroy();
                    mModule = null;
                }
                result.success(null);
                break;
            case "forward":
                try {
                    Tensor inputTensor = mapToTensor(call.arguments());
                    Tensor outputTensor = mModule.forward(IValue.from(inputTensor)).toTensor();
                    result.success(tensorToMap(outputTensor));
                } catch (Exception e) {
                    result.error("forwardError", "Pytorch lite module forward error.", e);
                }
                break;
            default:
                result.notImplemented();
                break;
        }
    }

    public static <T> int getJniCode(T t) {
        try {
            Field field = t.getClass().getDeclaredField("jniCode");
            field.setAccessible(true);
            return field.getInt(t);
        } catch (Exception ignored) {
        }
        return 0;
    }

    public static MemoryFormat parseMemoryFormat(Integer jniCode) {
        if (jniCode != null) {
            for (MemoryFormat memoryFormat : MemoryFormat.values()) {
                if (getJniCode(memoryFormat) == jniCode) return memoryFormat;
            }
        }
        throw new IllegalArgumentException(jniCode + " is not a valid code for MemoryFormat.");
    }

    public static DType parseDType(Integer jniCode) {
        if (jniCode != null) {
            for (DType dType : DType.values()) {
                if (getJniCode(dType) == jniCode) return dType;
            }
        }
        throw new IllegalArgumentException(jniCode + " is not a valid code for DType.");
    }

    Tensor mapToTensor(@Nullable HashMap<String, Object> map) {
        assert map != null;
        long[] shape = (long[]) map.get("shape");
        MemoryFormat memoryFormat = parseMemoryFormat((Integer) map.get("memoryFormat"));
        DType dtype = parseDType((Integer) map.get("dtype"));
        Object data = map.get("data");
        assert shape != null && data != null;
        switch (dtype) {
            case UINT8:
                return Tensor.fromBlobUnsigned((byte[]) data, shape, memoryFormat);
            case INT8:
                return Tensor.fromBlob((byte[]) data, shape, memoryFormat);
            case INT32:
                return Tensor.fromBlob((int[]) data, shape, memoryFormat);
            case FLOAT32:
                return Tensor.fromBlob((float[]) data, shape, memoryFormat);
            case INT64:
                return Tensor.fromBlob((long[]) data, shape, memoryFormat);
            case FLOAT64:
                return Tensor.fromBlob((double[]) data, shape, memoryFormat);
        }
        throw new IllegalArgumentException("Map to Tensor error.");
    }

    HashMap<String, Object> tensorToMap(Tensor tensor) {
        HashMap<String, Object> map = new HashMap<>();
        map.put("shape", tensor.shape());
        map.put("memoryFormat", getJniCode(tensor.memoryFormat()));
        map.put("dtype", getJniCode(tensor.dtype()));
        switch (tensor.dtype()) {
            case UINT8:
                map.put("data", tensor.getDataAsUnsignedByteArray());
                break;
            case INT8:
                map.put("data", tensor.getDataAsByteArray());
                break;
            case INT32:
                map.put("data", tensor.getDataAsIntArray());
                break;
            case FLOAT32:
                map.put("data", tensor.getDataAsFloatArray());
                break;
            case INT64:
                map.put("data", tensor.getDataAsLongArray());
                break;
            case FLOAT64:
                map.put("data", tensor.getDataAsDoubleArray());
                break;
        }
        return map;
    }

    @Override
    public void onDetachedFromEngine(@NonNull FlutterPluginBinding binding) {
        channel.setMethodCallHandler(null);
    }
}
