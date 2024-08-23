part of 'flutter_pytorch_lite.dart';

/// PyTorch Lite
///
/// @author guoweifeng
/// @date 2024/1/10 10:37

/// IValue
class IValue {
  static const int typeCodeNull = 1;
  static const int typeCodeTensor = 2;
  static const int typeCodeBool = 3;
  static const int typeCodeLong = 4;
  static const int typeCodeDouble = 5;
  static const int typeCodeString = 6;
  static const int typeCodeTuple = 7;
  static const int typeCodeBoolList = 8;
  static const int typeCodeLongList = 9;
  static const int typeCodeDoubleList = 10;
  static const int typeCodeTensorList = 11;
  static const int typeCodeList = 12;
  static const int typeCodeDictStringKey = 13;
  static const int typeCodeDictLongKey = 14;

  List<String> typeNames = [
    "Unknown",
    "Null",
    "Tensor",
    "Bool",
    "Long",
    "Double",
    "String",
    "Tuple",
    "BoolList",
    "LongList",
    "DoubleList",
    "TensorList",
    "GenericList",
    "DictStringKey",
    "DictLongKey",
  ];

  final int _typeCode;
  final _data;

  IValue._({required int typeCode, data})
      : _typeCode = typeCode,
        _data = data;

  bool get isNull => typeCodeNull == _typeCode;

  bool get isTensor => typeCodeTensor == _typeCode;

  bool get isBool => typeCodeBool == _typeCode;

  bool get isLong => typeCodeLong == _typeCode;

  bool get isDouble => typeCodeDouble == _typeCode;

  bool get isString => typeCodeString == _typeCode;

  bool get isTuple => typeCodeTuple == _typeCode;

  bool get isBoolList => typeCodeBoolList == _typeCode;

  bool get isLongList => typeCodeLongList == _typeCode;

  bool get isDoubleList => typeCodeDoubleList == _typeCode;

  bool get isTensorList => typeCodeTensorList == _typeCode;

  bool get isList => typeCodeList == _typeCode;

  bool get isDictStringKey => typeCodeDictStringKey == _typeCode;

  bool get isDictLongKey => typeCodeDictLongKey == _typeCode;

  Tensor toTensor() {
    _preconditionType(typeCodeTensor, _typeCode);
    return _data;
  }

  bool toBool() {
    _preconditionType(typeCodeBool, _typeCode);
    return _data;
  }

  int toLong() {
    _preconditionType(typeCodeLong, _typeCode);
    return _data;
  }

  double toDouble() {
    _preconditionType(typeCodeDouble, _typeCode);
    return _data;
  }

  String toStr() {
    _preconditionType(typeCodeString, _typeCode);
    return _data;
  }

  List<bool> toBoolList() {
    _preconditionType(typeCodeBoolList, _typeCode);
    return _data;
  }

  List<int> toLongList() {
    _preconditionType(typeCodeLongList, _typeCode);
    return _data;
  }

  List<double> toDoubleList() {
    _preconditionType(typeCodeDoubleList, _typeCode);
    return _data;
  }

  List<Tensor> toTensorList() {
    _preconditionType(typeCodeTensorList, _typeCode);
    return _data;
  }

  List<IValue> toList() {
    _preconditionType(typeCodeList, _typeCode);
    return _data;
  }

  List<IValue> toTuple() {
    _preconditionType(typeCodeTuple, _typeCode);
    return _data;
  }

  Map<String, IValue> toDictStringKey() {
    _preconditionType(typeCodeDictStringKey, _typeCode);
    return _data;
  }

  Map<int, IValue> toDictLongKey() {
    _preconditionType(typeCodeDictLongKey, _typeCode);
    return _data;
  }

  void _preconditionType(int typeCodeExpected, int typeCode) {
    if (typeCode != typeCodeExpected) {
      throw StateError(
          'Expected IValue type ${_getTypeName(typeCodeExpected)}, actual type ${_getTypeName(typeCode)}');
    }
  }

  String _getTypeName(int typeCode) {
    return typeCode >= 0 && typeCode < typeNames.length
        ? typeNames[typeCode]
        : "Unknown";
  }

  static IValue optionalNull() {
    return IValue._(typeCode: typeCodeNull);
  }

  static IValue from(value) {
    if (value is Tensor) {
      return IValue._(typeCode: typeCodeTensor, data: value);
    } else if (value is bool) {
      return IValue._(typeCode: typeCodeBool, data: value);
    } else if (value is int) {
      return IValue._(typeCode: typeCodeLong, data: value);
    } else if (value is double) {
      return IValue._(typeCode: typeCodeDouble, data: value);
    } else if (value is String) {
      return IValue._(typeCode: typeCodeString, data: value);
    } else if (value is List) {
      throw ArgumentError("Please use listFrom() or tupleFrom() instead");
    } else if (value is Map) {
      throw ArgumentError(
          "Please use dictStringKeyFrom() or dictLongKeyFrom() instead");
    }
    throw ArgumentError("Unsupported type ${value.runtimeType}");
  }

  static IValue listFrom(List list) {
    if (list is List<bool>) {
      return IValue._(typeCode: typeCodeBoolList, data: list);
    } else if (list is List<int>) {
      return IValue._(typeCode: typeCodeLongList, data: list);
    } else if (list is List<double>) {
      return IValue._(typeCode: typeCodeDoubleList, data: list);
    } else if (list is List<Tensor>) {
      return IValue._(typeCode: typeCodeTensorList, data: list);
    } else if (list is List<IValue>) {
      final size = list.length;
      if (size > 0) {
        final typeCode0 = list.first._typeCode;
        for (var i = 1; i < size; i++) {
          if (list[i]._typeCode != typeCode0) {
            throw ArgumentError("List must contain items of the same type");
          }
        }
      }
      return IValue._(typeCode: typeCodeList, data: list);
    }
    throw ArgumentError("Unsupported type ${list.runtimeType}");
  }

  static IValue tupleFrom(List<IValue> array) {
    return IValue._(typeCode: typeCodeTuple, data: array);
  }

  static IValue dictStringKeyFrom(Map<String, IValue> map) {
    return IValue._(typeCode: typeCodeDictStringKey, data: map);
  }

  static IValue dictLongKeyFrom(Map<int, IValue> map) {
    return IValue._(typeCode: typeCodeDictLongKey, data: map);
  }

  static IValue fromMap(map) {
    final typeCode = map['typeCode'];
    final data = map['data'];
    switch (typeCode) {
      case typeCodeNull:
        return optionalNull();
      case typeCodeTensor:
        return from(Tensor.fromMap(data));
      case typeCodeBool:
      case typeCodeLong:
      case typeCodeDouble:
      case typeCodeString:
        return from(data);
      case typeCodeTuple:
        return tupleFrom(data.map(fromMap).toList().cast<IValue>());
      case typeCodeBoolList:
      case typeCodeLongList:
      case typeCodeDoubleList:
        return listFrom(data);
      case typeCodeTensorList:
        return listFrom(data.map(Tensor.fromMap).toList().cast<Tensor>());
      case typeCodeList:
        return listFrom(data.map(fromMap).toList().cast<IValue>());
      case typeCodeDictStringKey:
        final Map<String, IValue> valueMap = {};
        data.forEach((key, value) {
          valueMap[key] = fromMap(value);
        });
        return dictStringKeyFrom(valueMap);
      case typeCodeDictLongKey:
        final Map<int, IValue> valueMap = {};
        data.forEach((key, value) {
          valueMap[key] = fromMap(value);
        });
        return dictLongKeyFrom(valueMap);
      default:
        throw ArgumentError("Unsupported type $typeCode");
    }
  }

  Map<String, dynamic> toMap() {
    Map<String, dynamic> map = {'typeCode': _typeCode};
    switch (_typeCode) {
      case typeCodeNull:
        break;
      case typeCodeTensor:
        map['data'] = _data.toMap();
        break;
      case typeCodeBool:
      case typeCodeLong:
      case typeCodeDouble:
      case typeCodeString:
        map['data'] = _data;
        break;
      case typeCodeBoolList:
      case typeCodeLongList:
        map['data'] = Int64List.fromList(_data);
        break;
      case typeCodeDoubleList:
        map['data'] = Float64List.fromList(_data);
        break;
      case typeCodeTensorList:
        map['data'] = _data.map((tensor) => tensor.toMap()).toList();
        break;
      case typeCodeTuple:
      case typeCodeList:
        map['data'] = _data.map((value) => value.toMap()).toList();
        break;
      case typeCodeDictStringKey:
      case typeCodeDictLongKey:
        final Map valueMap = {};
        _data.forEach((key, value) => valueMap[key] = value.toMap());
        map['data'] = valueMap;
        break;
      default:
        throw ArgumentError("Unsupported type $_typeCode");
    }
    return map;
  }
}

/// Memory format of tensor data.
enum MemoryFormat {
  contiguous(1),
  channelsLast(2),
  channelsLast3d(3),
  ;

  /// Code for memory format.
  final int jniCode;

  const MemoryFormat(this.jniCode);

  static MemoryFormat fromJniCode(int jniCode) {
    for (final format in values) {
      if (format.jniCode == jniCode) {
        return format;
      }
    }
    throw ArgumentError('Unknown MemoryFormat: $jniCode');
  }
}

/// Codes representing tensor data types.
enum DType {
  /// Code for dtype torch.uint8.
  uint8(1),

  /// Code for dtype torch.int8.
  int8(2),

  /// Code for dtype torch.int32.
  int32(3),

  /// Code for dtype torch.float32.
  float32(4),

  /// Code for dtype torch.int64.
  int64(5),

  /// Code for dtype torch.float64.
  float64(6),
  ;

  /// Code for dtype.
  final int jniCode;

  const DType(this.jniCode);

  static DType fromJniCode(int jniCode) {
    for (final type in values) {
      if (type.jniCode == jniCode) {
        return type;
      }
    }
    throw ArgumentError('Unknown DType: $jniCode');
  }
}

abstract class Tensor {
  final Int64List shape;
  final MemoryFormat memoryFormat;

  Tensor({required this.shape, this.memoryFormat = MemoryFormat.contiguous});

  DType get dtype => throw UnimplementedError();

  Uint8List get dataAsUin8List => throw UnimplementedError();

  Int8List get dataAsInt8List => throw UnimplementedError();

  Int32List get dataAsInt32List => throw UnimplementedError();

  Float32List get dataAsFloat32List => throw UnimplementedError();

  Int64List get dataAsInt64List => throw UnimplementedError();

  Float64List get dataAsFloat64List => throw UnimplementedError();

  static Tensor fromBlobUint8(Uint8List data, Int64List shape,
      {MemoryFormat memoryFormat = MemoryFormat.contiguous}) {
    return TensorUint8(data: data, shape: shape, memoryFormat: memoryFormat);
  }

  static Tensor fromBlobInt8(Int8List data, Int64List shape,
      {MemoryFormat memoryFormat = MemoryFormat.contiguous}) {
    return TensorInt8(data: data, shape: shape, memoryFormat: memoryFormat);
  }

  static Tensor fromBlobInt32(Int32List data, Int64List shape,
      {MemoryFormat memoryFormat = MemoryFormat.contiguous}) {
    return TensorInt32(data: data, shape: shape, memoryFormat: memoryFormat);
  }

  static Tensor fromBlobFloat32(Float32List data, Int64List shape,
      {MemoryFormat memoryFormat = MemoryFormat.contiguous}) {
    return TensorFloat32(data: data, shape: shape, memoryFormat: memoryFormat);
  }

  static Tensor fromBlobInt64(Int64List data, Int64List shape,
      {MemoryFormat memoryFormat = MemoryFormat.contiguous}) {
    return TensorInt64(data: data, shape: shape, memoryFormat: memoryFormat);
  }

  static Tensor fromBlobFloat64(Float64List data, Int64List shape,
      {MemoryFormat memoryFormat = MemoryFormat.contiguous}) {
    return TensorFloat64(data: data, shape: shape, memoryFormat: memoryFormat);
  }

  static Tensor fromMap(map) {
    Int64List shape = map['shape'];
    MemoryFormat memoryFormat = MemoryFormat.fromJniCode(map['memoryFormat']);
    switch (DType.fromJniCode(map['dtype'])) {
      case DType.uint8:
        return fromBlobUint8(map['data'], shape, memoryFormat: memoryFormat);
      case DType.int8:
        return fromBlobInt8(map['data'], shape, memoryFormat: memoryFormat);
      case DType.int32:
        return fromBlobInt32(map['data'], shape, memoryFormat: memoryFormat);
      case DType.float32:
        return fromBlobFloat32(map['data'], shape, memoryFormat: memoryFormat);
      case DType.int64:
        return fromBlobInt64(map['data'], shape, memoryFormat: memoryFormat);
      case DType.float64:
        return fromBlobFloat64(map['data'], shape, memoryFormat: memoryFormat);
    }
  }

  Map<String, dynamic> toMap() {
    Map<String, dynamic> map = {
      'shape': shape,
      'memoryFormat': memoryFormat.jniCode,
      'dtype': dtype.jniCode,
    };
    switch (dtype) {
      case DType.uint8:
        map['data'] = dataAsUin8List;
        break;
      case DType.int8:
        map['data'] = dataAsInt8List;
        break;
      case DType.int32:
        map['data'] = dataAsInt32List;
        break;
      case DType.float32:
        map['data'] = dataAsFloat32List;
        break;
      case DType.int64:
        map['data'] = dataAsInt64List;
        break;
      case DType.float64:
        map['data'] = dataAsFloat64List;
        break;
    }
    return map;
  }
}

class TensorUint8 extends Tensor {
  final Uint8List data;

  TensorUint8({required super.shape, super.memoryFormat, required this.data});

  @override
  DType get dtype => DType.uint8;

  @override
  Uint8List get dataAsUin8List => data;
}

class TensorInt8 extends Tensor {
  final Int8List data;

  TensorInt8({required super.shape, super.memoryFormat, required this.data});

  @override
  DType get dtype => DType.int8;

  @override
  Int8List get dataAsInt8List => data;
}

class TensorInt32 extends Tensor {
  final Int32List data;

  TensorInt32({required super.shape, super.memoryFormat, required this.data});

  @override
  DType get dtype => DType.int32;

  @override
  Int32List get dataAsInt32List => data;
}

class TensorFloat32 extends Tensor {
  final Float32List data;

  TensorFloat32({required super.shape, super.memoryFormat, required this.data});

  @override
  DType get dtype => DType.float32;

  @override
  Float32List get dataAsFloat32List => data;
}

class TensorInt64 extends Tensor {
  final Int64List data;

  TensorInt64({required super.shape, super.memoryFormat, required this.data});

  @override
  DType get dtype => DType.int64;

  @override
  Int64List get dataAsInt64List => data;
}

class TensorFloat64 extends Tensor {
  final Float64List data;

  TensorFloat64({required super.shape, super.memoryFormat, required this.data});

  @override
  DType get dtype => DType.float64;

  @override
  Float64List get dataAsFloat64List => data;
}
