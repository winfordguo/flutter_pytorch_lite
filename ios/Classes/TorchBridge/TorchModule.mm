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
        @throw [NSException exceptionWithName:NSInvalidArgumentException
                                       reason:@"Map to Tensor error."
                                     userInfo:nil];
    }
    // return at::Tensor();  // 返回一个空张量表示出错
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
        @throw [NSException exceptionWithName:NSInvalidArgumentException
                                       reason:@"Tensor to Map error."
                                     userInfo:nil];
    }
    // return nil;
}

- (at::IValue)mapToIValue:(NSDictionary *)map {
    try {
        NSInteger typeCode = [map[@"typeCode"] integerValue];
        //auto *data = map[@"data"];

        switch (typeCode) {
            case kTypeCodeNull:
                return at::IValue{};
            case kTypeCodeTensor: {
                NSDictionary *data = map[@"data"];
                return at::IValue{[self mapToTensor:data]};
            }
            case kTypeCodeBool: {
                NSNumber *data = map[@"data"];
                return at::IValue{[data boolValue]};
            }
            case kTypeCodeLong: {
                NSNumber *data = map[@"data"];
                return at::IValue{[data longLongValue]};
            }
            case kTypeCodeDouble: {
                NSNumber *data = map[@"data"];
                return at::IValue{[data doubleValue]};
            }
            case kTypeCodeString: {
                NSString *data = map[@"data"];
                return at::IValue{[data cStringUsingEncoding:NSUTF8StringEncoding]};
            }
            case kTypeCodeTuple: {
                NSArray *data = map[@"data"];
                size_t n = [data count];

                std::vector<at::IValue> array;
                array.reserve(n);
                for (NSDictionary *item in data) {
                    array.push_back(std::move([self mapToIValue:item]));
                }
                return c10::ivalue::Tuple::create(std::move(array));
            }
            case kTypeCodeBoolList: {
                NSArray *data = map[@"data"];
                size_t n = [data count];

                c10::List<bool> list{};
                list.reserve(n);
                for (NSNumber *item in data) {
                    list.push_back([item boolValue]);
                }
                return at::IValue(std::move(list));
            }
            case kTypeCodeLongList: {
                NSArray *data = map[@"data"];
                size_t n = [data count];

                c10::List <int64_t> list{};
                list.reserve(n);
                for (NSNumber *item in data) {
                    list.push_back([item longValue]);
                }
                return at::IValue{std::move(list)};
            }
            case kTypeCodeDoubleList: {
                NSArray *data = map[@"data"];
                size_t n = [data count];

                c10::List<double> list{};
                list.reserve(n);
                for (NSNumber *item in data) {
                    list.push_back([item doubleValue]);
                }
                return at::IValue{std::move(list)};
            }
            case kTypeCodeTensorList: {
                NSArray *data = map[@"data"];
                size_t n = [data count];

                c10::List<at::Tensor> list{};
                list.reserve(n);
                for (NSDictionary *item in data) {
                    list.push_back([self mapToTensor:item]);
                }
                return at::IValue{std::move(list)};
            }
            case kTypeCodeList: {
                NSArray *data = map[@"data"];
                size_t n = [data count];
                if (n == 0) {
                  return at::IValue{c10::impl::GenericList(c10::TensorType::get())};
                }

                c10::impl::GenericList list{at::IValue{}.type()};
                list.reserve(n);
                for (NSDictionary *item in data) {
                    list.push_back([self mapToIValue:item]);
                }
                return at::IValue{std::move(list)};
            }
            case kTypeCodeDictStringKey: {
                NSDictionary *data = map[@"data"];
                if (data.count == 0) {
                  return at::IValue{c10::impl::GenericDict(c10::StringType::get(), c10::TensorType::get())};
                }

                c10::impl::GenericDict genericDict{c10::StringType::get(), at::IValue{}.type()};
                for (NSString *key in data.allKeys) {
                    genericDict.insert(std::string([key UTF8String]), [self mapToIValue:data[key]]);
                }
                return at::IValue{genericDict};
            }
            case kTypeCodeDictLongKey: {
                NSDictionary *data = map[@"data"];
                if (data.count == 0) {
                  return at::IValue{c10::impl::GenericDict(c10::IntType::get(), c10::TensorType::get())};
                }

                c10::impl::GenericDict genericDict{c10::IntType::get(), at::IValue{}.type()};
                for (NSNumber *key in data.allKeys) {
                    genericDict.insert([key intValue], [self mapToIValue:data[key]]);
                }
                return at::IValue{genericDict};
            }
        }
        @throw [NSException exceptionWithName:NSInvalidArgumentException
                                       reason:[NSString stringWithFormat:@"Unknown IValue typeCode %@", typeCode]
                                     userInfo:nil];
    } catch (const std::exception& exception) {
        NSLog(@"Error: %s", exception.what());
        @throw [NSException exceptionWithName:NSInvalidArgumentException
                                       reason:@"Map to IValue error."
                                     userInfo:nil];
    }
}

- (NSDictionary *)iValueToMap:(at::IValue)value {
    try {
        if (value.isNone()) {
            return @{
                    @"typeCode": @(kTypeCodeNull),
            };
        } else if (value.isTensor()) {
            return @{
                    @"typeCode": @(kTypeCodeTensor),
                    @"data": [self tensorToMap:value.toTensor()]
            };
        } else if (value.isBool()) {
            return @{
                    @"typeCode": @(kTypeCodeBool),
                    @"data": @(value.toBool())
            };
        } else if (value.isDouble()) {
            return @{
                    @"typeCode": @(kTypeCodeDouble),
                    @"data": @(value.toDouble())
            };
        } else if (value.isInt()) {
            return @{
                    @"typeCode": @(kTypeCodeLong),
                    @"data": @(value.toInt())
            };
        } else if (value.isString()) {
            return @{
                    @"typeCode": @(kTypeCodeString),
                    @"data": [NSString stringWithUTF8String:value.toStringRef().c_str()]
            };
        } else if (value.isTuple()) {
            auto array = value.toTuple()->elements();
            NSMutableArray *list = [NSMutableArray arrayWithCapacity:array.size()];
            for (const auto& item : array) {
                [list addObject:[self iValueToMap:item]];
            }
            return @{
                    @"typeCode": @(kTypeCodeTuple),
                    @"data": list
            };
        } else if (value.isBoolList()) {
            c10::List<bool> array = value.toBoolList();
            NSMutableArray *list = [NSMutableArray arrayWithCapacity:array.size()];
            for (bool item : array) {
                [list addObject:@(item)];
            }
            return @{
                    @"typeCode": @(kTypeCodeBoolList),
                    @"data": list
            };
        } else if (value.isIntList()) {
            c10::List<int64_t> array = value.toIntList();
            NSMutableArray *list = [NSMutableArray arrayWithCapacity:array.size()];
            for (int64_t item : array) {
                [list addObject:@(item)];
            }
            return @{
                    @"typeCode": @(kTypeCodeLongList),
                    @"data": list
            };
        } else if (value.isDoubleList()) {
            c10::List<double> array = value.toDoubleList();
            NSMutableArray *list = [NSMutableArray arrayWithCapacity:array.size()];
            for (double item : array) {
                [list addObject:@(item)];
            }
            return @{
                    @"typeCode": @(kTypeCodeDoubleList),
                    @"data": list
            };
        } else if (value.isTensorList()) {
            c10::List<at::Tensor> array = value.toTensorList();
            NSMutableArray *list = [NSMutableArray arrayWithCapacity:array.size()];
            for (at::Tensor item : array) {
                [list addObject:[self tensorToMap:item]];
            }
            return @{
                    @"typeCode": @(kTypeCodeTensorList),
                    @"data": list
            };
        } else if (value.isList()) {
            auto array = value.toList();
            NSMutableArray *list = [NSMutableArray arrayWithCapacity:array.size()];
            for (const auto& item : array) {
                [list addObject:[self iValueToMap:item]];
            }
            return @{
                    @"typeCode": [NSNumber numberWithInt:kTypeCodeList],
                    @"data": list
            };
        } else if (value.isGenericDict()) {
            auto dict = value.toGenericDict();
            NSMutableDictionary *map = [NSMutableDictionary dictionaryWithCapacity:dict.size()];

            const auto keyType = dict.keyType();
            if (*keyType == *c10::StringType::get()) {
                for (auto &item : dict) {
                    map[@(item.key().toStringRef().c_str())] = [self iValueToMap:item.value()];
                }
                return @{
                        @"typeCode": @(kTypeCodeDictStringKey),
                        @"data": map
                };
            } else if (*keyType == *c10::IntType::get()) {
                for (auto &item : dict) {
                    map[@(item.key().toInt())] = [self iValueToMap:item.value()];
                }
                return @{
                        @"typeCode": @(kTypeCodeDictLongKey),
                        @"data": map
                };
            } else {
               @throw [NSException exceptionWithName:NSInvalidArgumentException 
                                              reason:[NSString stringWithFormat:@"Unsupported IValue-Dict key type %@", [NSString stringWithUTF8String:keyType->str().c_str()]]
                                            userInfo:nil];
            }
        } else {
//            @throw [NSException exceptionWithName:NSInvalidArgumentException
//                                           reason:[NSString stringWithFormat:@"Unsupported IValue type %@", [NSString stringWithUTF8String:value.tagKind.c_str()]]
//                                         userInfo:nil];
        }
    } catch (const std::exception& exception) {
        NSLog(@"Error: %s", exception.what());
        @throw [NSException exceptionWithName:NSInvalidArgumentException
                                       reason:@"IValue to Map error."
                                     userInfo:nil];
    }
     return nil;
}

- (NSDictionary *)forwardArray:(NSArray *)array {
    try {
        std::vector<c10::IValue> inputs = {};
        for (int i = 0; i < array.count; i++) {
            at::IValue input = [self mapToIValue: array[i]];
            inputs.push_back(input);
        }
        auto outputs = _impl.forward(inputs);
        return [self iValueToMap: outputs];
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}

@end
