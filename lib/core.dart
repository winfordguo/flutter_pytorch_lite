part of 'flutter_pytorch_lite.dart';

/// PyTorch Lite
///
/// @author guoweifeng
/// @date 2024/1/10 10:37

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
