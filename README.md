# Flutter PyTorch Lite

<p>
  <a href="https://flutter.dev">
    <img src="https://img.shields.io/badge/Platform-Flutter-02569B?logo=flutter" alt="Platform" /></a>
  <a href="https://pub.dev/packages/flutter_pytorch_lite">
    <img src="https://img.shields.io/pub/v/flutter_pytorch_lite.svg" alt="Pub Package" /></a>
  <a href="https://pub.dev/documentation/flutter_pytorch_lite/latest/flutter_pytorch_lite/flutter_pytorch_lite-library.html">
    <img src="https://readthedocs.org/projects/hubdb/badge/?version=latest" alt="Docs" /></a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License" /></a>
</p>

PyTorch Lite plugin for Flutter.

End-to-end workflow from Training to Deployment for iOS and Android mobile devices.

## [PyTorch Mobile](https://pytorch.org/mobile/home/)
There is a growing need to execute ML models on edge devices to reduce latency, preserve privacy, and enable new interactive use cases.

The PyTorch Mobile runtime beta release allows you to seamlessly go from training a model to deploying it, while staying entirely within the PyTorch ecosystem. It provides an end-to-end workflow that simplifies the research to production environment for mobile devices. In addition, it paves the way for privacy-preserving features via federated learning techniques.

PyTorch Mobile is in beta stage right now, and is already in wide scale production use. It will soon be available as a stable release once the APIs are locked down.

## Requirements

* ### Android
  * minSdkVersion 21

* ### iOS
  * XCode 11.0 or above
  * iOS 12.0 or above

## Usage instructions

### Install

In the dependency section of `pubspec.yaml` file, add `flutter_pytorch_lite` (adjust the version accordingly based on the latest release)

```yaml
dependencies:
  flutter_pytorch_lite: ^0.0.1+1
```
or
```yaml
dependencies:
  flutter_pytorch_lite:
    git:
      url: https://github.com/winfordguo/flutter_pytorch_lite.git
```

### Import

```dart
import 'package:flutter_pytorch_lite/flutter_pytorch_lite.dart';
```

### Loading the model

* **From path**

    ```dart
    await FlutterPytorchLite.load('/path/to/your_model.ptl');
    ```

* **From asset**

  Place `your_model.ptl` in `assets` directory. Make sure to include assets in `pubspec.yaml`.

    ```dart
    final filePath = '${Directory.systemTemp.path}/your_model.ptl';
    File(filePath).writeAsBytesSync(await _getBuffer('assets/your_model.ptl'));
    await FlutterPytorchLite.load(filePath);
  
    /// Get byte buffer
    static Future<Uint8List> _getBuffer(String assetFileName) async {
      ByteData rawAssetFile = await rootBundle.load(assetFileName);
      final rawBytes = rawAssetFile.buffer.asUint8List();
      return rawBytes;
    }
    ```

Refer to the documentation for info on creating interpreter from buffer or file.

### Forwarding

* **For single input and output**

  Use `static Tensor forward(Tensor input)`.
    ```dart
    // For ex: if input tensor shape [1,5] and type is float32
    final inputShape = Int64List.fromList([1, 5]);
    var input = [1.23, 6.54, 7.81, 3.21, 2.22];
    Tensor inputTensor = Tensor.fromBlobFloat32(input, inputShape);

    // forward
    Tensor outputTensor = await FlutterPytorchLite.forward(inputTensor);

    // Get output tensor: if output tensor type is float32
    final outputShape = outputTensor.shape;
    var output = outputTensor.dataAsFloat32List;

    // print the output
    print(output);
    ```

### Destroying the model

```dart
await FlutterPytorchLite.destroy();
```

## Q&A

### Android

* **Q:** Execution failed for task ':app:mergeDebugNativeLibs'

  ```
  * What went wrong:
  Execution failed for task ':app:mergeDebugNativeLibs'.
  > A failure occurred while executing com.android.build.gradle.internal.tasks.Workers$ActionFacade
     > More than one file was found with OS independent path 'lib/x86/libc++_shared.so'
  ```

  **A:** add this to your `app/build.gradle`

  ```groovy
  android {
     // your existing code
     packagingOptions {
          pickFirst '**/libc++_shared.so'
      }
  }
  ```

* **Q:** What is the version of PyTorch Lite?

  **A:** `org.pytorch:pytorch_android_lite:1.10.0` and `org.pytorch:pytorch_android_torchvision_lite:1.10.0`

### iOS

* **Q:** What is the version of PyTorch Lite?

  **A:** `'LibTorch-Lite', '~> 1.10.0'`

