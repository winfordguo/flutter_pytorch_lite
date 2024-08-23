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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

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

    private static final ConcurrentHashMap<Integer, Module> mModules = new ConcurrentHashMap<>();

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
                    Module mModule = LiteModuleLoader.load(filePath);
                    mModules.put(mModule.hashCode(), mModule);
                    result.success(mModule.hashCode());
                } catch (Exception e) {
                    result.error("loadError", "Pytorch lite load module error", e);
                }
                break;
            case "destroy": {
                Integer moduleId = call.argument("moduleId");
                assert moduleId != null;
                Module mModule = mModules.remove(moduleId);
                if (mModule != null) {
                    mModule.destroy();
                }
                result.success(null);
                break;
            }
            case "forward": {
                try {
                    Integer moduleId = call.argument("moduleId");
                    assert moduleId != null;
                    Module mModule = mModules.get(moduleId);

                    List<HashMap<String, Object>> inputList = call.argument("inputs");
                    assert inputList != null && !inputList.isEmpty();
                    IValue[] inputs = new IValue[inputList.size()];
                    for (int i = 0; i < inputs.length; i++) {
                        inputs[i] = mapToIValue(inputList.get(i));
                    }
                    IValue outputs = mModule.forward(inputs);
                    result.success(iValueToMap(outputs));
                } catch (Exception e) {
                    result.error("forwardError", "Pytorch lite module forward error.", e);
                }
                break;
            }
            default:
                result.notImplemented();
                break;
        }
    }

    @Override
    public void onDetachedFromEngine(@NonNull FlutterPluginBinding binding) {
        channel.setMethodCallHandler(null);
        try {
            for (Map.Entry<Integer, Module> entry : mModules.entrySet()) {
                entry.getValue().destroy();
            }
            mModules.clear();
        } catch (Exception ignore) {
        }
    }

    private static final int TYPE_CODE_NULL = 1;
    private static final int TYPE_CODE_TENSOR = 2;
    private static final int TYPE_CODE_BOOL = 3;
    private static final int TYPE_CODE_LONG = 4;
    private static final int TYPE_CODE_DOUBLE = 5;
    private static final int TYPE_CODE_STRING = 6;

    private static final int TYPE_CODE_TUPLE = 7;
    private static final int TYPE_CODE_BOOL_LIST = 8;
    private static final int TYPE_CODE_LONG_LIST = 9;
    private static final int TYPE_CODE_DOUBLE_LIST = 10;
    private static final int TYPE_CODE_TENSOR_LIST = 11;
    private static final int TYPE_CODE_LIST = 12;

    private static final int TYPE_CODE_DICT_STRING_KEY = 13;
    private static final int TYPE_CODE_DICT_LONG_KEY = 14;

    IValue mapToIValue(@Nullable HashMap<String, Object> map) {
        assert map != null;

        Integer typeCode = (Integer) map.get("typeCode");
        assert typeCode != null;

        if (typeCode == TYPE_CODE_NULL) return IValue.optionalNull();

        Object data = map.get("data");
        assert data != null;
        switch (typeCode) {
            case TYPE_CODE_TENSOR:
                return IValue.from(mapToTensor((HashMap<String, Object>) data));
            case TYPE_CODE_BOOL:
                return IValue.from((boolean) data);
            case TYPE_CODE_LONG:
                return IValue.from((long) data);
            case TYPE_CODE_DOUBLE:
                return IValue.from((double) data);
            case TYPE_CODE_STRING:
                return IValue.from((String) data);
            case TYPE_CODE_TUPLE: {
                ArrayList list = (ArrayList) data;
                IValue[] array = new IValue[list.size()];
                for (int i = 0; i < array.length; i++) {
                    array[i] = mapToIValue((HashMap<String, Object>) list.get(i));
                }
                return IValue.tupleFrom(array);
            }
            case TYPE_CODE_BOOL_LIST: {
                ArrayList list = (ArrayList) data;
                boolean[] array = new boolean[list.size()];
                for (int i = 0; i < array.length; i++) {
                    array[i] = (boolean) list.get(i);
                }
                return IValue.listFrom(array);
            }
            case TYPE_CODE_LONG_LIST:
                return IValue.listFrom((long[]) data);
            case TYPE_CODE_DOUBLE_LIST:
                return IValue.listFrom((double[]) data);
            case TYPE_CODE_TENSOR_LIST: {
                ArrayList list = (ArrayList) data;
                Tensor[] array = new Tensor[list.size()];
                for (int i = 0; i < array.length; i++) {
                    array[i] = mapToTensor((HashMap<String, Object>) list.get(i));
                }
                return IValue.listFrom(array);
            }
            case TYPE_CODE_LIST: {
                ArrayList list = (ArrayList) data;
                IValue[] array = new IValue[list.size()];
                for (int i = 0; i < array.length; i++) {
                    array[i] = mapToIValue((HashMap<String, Object>) list.get(i));
                }
                return IValue.listFrom(array);
            }
            case TYPE_CODE_DICT_STRING_KEY: {
                HashMap<String, Object> dict = (HashMap<String, Object>) data;
                HashMap<String, IValue> valueMap = new HashMap<>();
                for (String key : dict.keySet()) {
                    valueMap.put(key, mapToIValue((HashMap<String, Object>) dict.get(key)));
                }
                return IValue.dictStringKeyFrom(valueMap);
            }
            case TYPE_CODE_DICT_LONG_KEY: {
                HashMap<Long, Object> dict = (HashMap<Long, Object>) data;
                HashMap<Long, IValue> valueMap = new HashMap<>();
                for (Long key : dict.keySet()) {
                    valueMap.put(key, mapToIValue((HashMap<String, Object>) dict.get(key)));
                }
                return IValue.dictLongKeyFrom(valueMap);
            }
        }
        throw new IllegalArgumentException("Map to IValue error.");
    }

    public static int getTypeCode(IValue value) {
        try {
            Field field = value.getClass().getDeclaredField("mTypeCode");
            field.setAccessible(true);
            return field.getInt(value);
        } catch (Exception ignored) {
        }
        return 0;
    }

    HashMap<String, Object> iValueToMap(IValue value) {
        int typeCode = getTypeCode(value);

        HashMap<String, Object> map = new HashMap<>();
        map.put("typeCode", typeCode);
        switch (typeCode) {
            case TYPE_CODE_NULL:
                break;
            case TYPE_CODE_TENSOR:
                map.put("data", tensorToMap(value.toTensor()));
                break;
            case TYPE_CODE_BOOL:
                map.put("data", value.toBool());
                break;
            case TYPE_CODE_LONG:
                map.put("data", value.toLong());
                break;
            case TYPE_CODE_DOUBLE:
                map.put("data", value.toDouble());
                break;
            case TYPE_CODE_STRING:
                map.put("data", value.toStr());
                break;
            case TYPE_CODE_TUPLE: {
                IValue[] array = value.toTuple();
                ArrayList<HashMap<String, Object>> list = new ArrayList<>();
                for (IValue iValue : array) {
                    list.add(iValueToMap(iValue));
                }
                map.put("data", list);
                break;
            }
            case TYPE_CODE_BOOL_LIST: {
                boolean[] array = value.toBoolList();
                ArrayList<Boolean> list = new ArrayList<>();
                for (Boolean e : array) list.add(e);
                map.put("data", list);
                break;
            }
            case TYPE_CODE_LONG_LIST:
                map.put("data", value.toLongList());
                break;
            case TYPE_CODE_DOUBLE_LIST:
                map.put("data", value.toDoubleList());
                break;
            case TYPE_CODE_TENSOR_LIST: {
                Tensor[] array = value.toTensorList();
                ArrayList<HashMap<String, Object>> list = new ArrayList<>();
                for (Tensor tensor : array) {
                    list.add(tensorToMap(tensor));
                }
                map.put("data", list);
                break;
            }
            case TYPE_CODE_LIST: {
                IValue[] array = value.toList();
                ArrayList<HashMap<String, Object>> list = new ArrayList<>();
                for (IValue iValue : array) {
                    list.add(iValueToMap(iValue));
                }
                map.put("data", list);
                break;
            }
            case TYPE_CODE_DICT_STRING_KEY: {
                Map<String, IValue> dict = value.toDictStringKey();
                HashMap<String, Object> valueMap = new HashMap<>();
                for (String key : dict.keySet()) {
                    valueMap.put(key, iValueToMap(dict.get(key)));
                }
                map.put("data", valueMap);
                break;
            }
            case TYPE_CODE_DICT_LONG_KEY: {
                Map<Long, IValue> dict = value.toDictLongKey();
                HashMap<Long, Object> valueMap = new HashMap<>();
                for (Long key : dict.keySet()) {
                    valueMap.put(key, iValueToMap(dict.get(key)));
                }
                map.put("data", valueMap);
                break;
            }
        }
        return map;
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
}
