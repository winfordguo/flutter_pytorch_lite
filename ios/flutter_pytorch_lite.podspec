#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint flutter_pytorch_lite.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'flutter_pytorch_lite'
  s.version          = '0.0.1'
  s.summary          = 'PyTorch Lite plugin for Flutter.'
  s.description      = <<-DESC
  PyTorch Lite plugin for Flutter.
  DESC
  s.homepage         = 'https://github.com/winfordguo/flutter_pytorch_lite'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Winford' => 'winfordguo@gmail.com' }
  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.static_framework = true
  s.dependency 'Flutter'
  # Pytorch Lite
  s.dependency 'LibTorch-Lite', '~> 1.13.0.1'
  s.platform = :ios, '12.0'
  
  # Flutter.framework does not contain a i386 slice.
  #s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386', 'HEADER_SEARCH_PATHS' => '"${PODS_ROOT}/LibTorch-Lite/install/include"' }
  s.swift_version = '5.0'
end
