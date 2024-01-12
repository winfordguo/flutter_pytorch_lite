#import "TorchModule.h"
#import <Libtorch-Lite/Libtorch-Lite.h>
#import <Flutter/Flutter.h>

/**
 * PyTorch Lite
 *
 * @author guoweifeng
 * @date 2024/1/10 10:37
 */
@implementation TorchModule {
@protected
    torch::jit::mobile::Module _impl;
}

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
    self = [super init];
    if (self) {
        try {
            _impl = torch::jit::_load_for_mobile(filePath.UTF8String);
        } catch (const std::exception& exception) {
            NSLog(@"%s", exception.what());
            return nil;
        }
    }
    return self;
}

- (void)destroy {
}

- (c10::optional<c10::ScalarType>)parseDType:(NSInteger)dtype {
    switch (dtype) {
        case 1: // UINT8
            return c10::kByte;
        case 2: // INT8
            return c10::kChar;
        case 3: // INT32
            return c10::kInt;
        case 4: // FLOAT32
            return c10::kFloat;
        case 5: // INT64
            return c10::kLong;
        case 6: // FLOAT64
            return c10::kDouble;
    }
    return c10::nullopt;
}

- (c10::optional<c10::MemoryFormat>)parseMemoryFormat:(NSInteger)memoryFormat {
    switch (memoryFormat) {
        case 1:
            return c10::MemoryFormat::Contiguous;
        case 2:
            return c10::MemoryFormat::ChannelsLast;
        case 3:
            return c10::MemoryFormat::ChannelsLast3d;
    }
    return c10::nullopt;
}

- (NSInteger)dtypeJniCode:(c10::ScalarType)dtype {
    switch (dtype) {
        case c10::kByte:
            return 1; // 对应 UINT8
        case c10::kChar:
            return 2; // 对应 INT8
        case c10::kInt:
            return 3; // 对应 INT32
        case c10::kFloat:
            return 4; // 对应 FLOAT32
        case c10::kLong:
            return 5; // 对应 INT64
        case c10::kDouble:
            return 6; // 对应 FLOAT64
        default:
            return 0; // 未知类型，可以根据实际情况返回其他值
    }
    return 0; // nil 对应 0
}

- (NSInteger)memoryFormatJniCode:(c10::MemoryFormat)memoryFormat {
    switch (memoryFormat) {
        case c10::MemoryFormat::Contiguous:
            return 1;
        case c10::MemoryFormat::ChannelsLast:
            return 2;
        case c10::MemoryFormat::ChannelsLast3d:
            return 3;
        default:
            return 0; // 未知内存格式，可以根据实际情况返回其他值
    }
    return 0; // nil 对应 0
}

- (at::Tensor)mapToTensor:(NSDictionary *)map {
    try {
        NSInteger dtype = [map[@"dtype"] integerValue];
        NSInteger memoryFormat = [map[@"memoryFormat"] integerValue];
        FlutterStandardTypedData *shapeData = map[@"shape"];
        FlutterStandardTypedData *data = map[@"data"];
        
        // 类型检查，确保 shapeData 和 data 是正确的类型
        if (![shapeData isKindOfClass:[FlutterStandardTypedData class]] || ![data isKindOfClass:[FlutterStandardTypedData class]]) {
            NSLog(@"Error: Invalid data type for shape or data.");
            return at::Tensor();  // 返回一个空张量表示出错
        }
        
        // 将 shapeData 转换为整数数组
        const int64_t *shapeArray = reinterpret_cast<const int64_t *>([shapeData.data bytes]);
        
        // 创建张量选项
        const at::TensorOptions options = at::TensorOptions()
            .dtype([self parseDType:dtype])
            .memory_format([self parseMemoryFormat:memoryFormat]);
        
        // 计算元素个数
        int64_t elementCount = data.elementCount;
        
        // 使用 torch::from_blob 创建张量
        at::Tensor tensor = torch::from_blob(const_cast<void *>(data.data.bytes), {elementCount}, options);
        
        // 调整张量的形状
        tensor = tensor.view(at::ArrayRef<int64_t>(shapeArray, [shapeData elementCount]));
        
        return tensor;
    } catch (const std::exception& exception) {
        NSLog(@"Error: %s", exception.what());
    }
    return at::Tensor();  // 返回一个空张量表示出错
}

- (NSDictionary *)tensorToMap:(at::Tensor)tensor {
    try {
        // 获取张量的数据类型和内存格式
        c10::ScalarType dtype = tensor.scalar_type();
        c10::MemoryFormat memoryFormat = tensor.suggest_memory_format();
        
        // 获取张量的形状和数据
        std::vector<int64_t> shapeVector(tensor.sizes().begin(), tensor.sizes().end());
        NSData *shapeData = [NSData dataWithBytes:shapeVector.data() length:shapeVector.size() * sizeof(int64_t)];
        FlutterStandardTypedData *shape = [FlutterStandardTypedData typedDataWithInt64: shapeData];
        
        FlutterStandardTypedData *data;
        switch (dtype) {
            case c10::kByte: {
                // UINT8
                std::vector<uint8_t> dataVector(tensor.data_ptr<uint8_t>(), tensor.data_ptr<uint8_t>() + tensor.numel());
                NSData *dataData = [NSData dataWithBytes:dataVector.data() length:dataVector.size() * sizeof(uint8_t)];
                data = [FlutterStandardTypedData typedDataWithBytes: dataData];
                break;
            }
            case c10::kChar: {
                // INT8
                std::vector<int8_t> dataVector(tensor.data_ptr<int8_t>(), tensor.data_ptr<int8_t>() + tensor.numel());
                NSData *dataData = [NSData dataWithBytes:dataVector.data() length:dataVector.size() * sizeof(int8_t)];
                data = [FlutterStandardTypedData typedDataWithInt32: dataData];
                break;
            }
            case c10::kInt: {
                // INT32
                std::vector<int32_t> dataVector(tensor.data_ptr<int32_t>(), tensor.data_ptr<int32_t>() + tensor.numel());
                NSData *dataData = [NSData dataWithBytes:dataVector.data() length:dataVector.size() * sizeof(int32_t)];
                data = [FlutterStandardTypedData typedDataWithInt32: dataData];
                break;
            }
            case c10::kFloat: {
                // FLOAT32
                std::vector<float> dataVector(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
                NSData *dataData = [NSData dataWithBytes:dataVector.data() length:dataVector.size() * sizeof(float)];
                data = [FlutterStandardTypedData typedDataWithFloat32: dataData];
                break;
            }
            case c10::kLong: {
                // INT64
                std::vector<int64_t> dataVector(tensor.data_ptr<int64_t>(), tensor.data_ptr<int64_t>() + tensor.numel());
                NSData *dataData = [NSData dataWithBytes:dataVector.data() length:dataVector.size() * sizeof(int64_t)];
                data = [FlutterStandardTypedData typedDataWithInt64: dataData];
                break;
            }
            case c10::kDouble: {
                // FLOAT64
                std::vector<double> dataVector(tensor.data_ptr<double>(), tensor.data_ptr<double>() + tensor.numel());
                NSData *dataData = [NSData dataWithBytes:dataVector.data() length:dataVector.size() * sizeof(double)];
                data = [FlutterStandardTypedData typedDataWithFloat64: dataData];
                break;
            }
            default:
                data = [FlutterStandardTypedData typedDataWithBytes: [NSData new]];
                break;
        }
 
        // 构建字典
        NSDictionary *tensorMap = @{
            @"dtype": @([self dtypeJniCode: dtype]),
            @"memoryFormat": @([self memoryFormatJniCode: memoryFormat]),
            @"shape": shape,
            @"data": data
        };
        
        return tensorMap;
    } catch (const std::exception& exception) {
        NSLog(@"Error: %s", exception.what());
    }
    return nil;
}

- (NSDictionary *)forwardMap:(NSDictionary *)map {
    try {
        at::Tensor inputTensor = [self mapToTensor: map];
        auto outputTensor = _impl.forward({inputTensor}).toTensor();
        return [self tensorToMap: outputTensor];
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}

@end
