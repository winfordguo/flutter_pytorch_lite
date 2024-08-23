#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

static const int kTypeCodeNull = 1;
static const int kTypeCodeTensor = 2;
static const int kTypeCodeBool = 3;
static const int kTypeCodeLong = 4;
static const int kTypeCodeDouble = 5;
static const int kTypeCodeString = 6;

static const int kTypeCodeTuple = 7;
static const int kTypeCodeBoolList = 8;
static const int kTypeCodeLongList = 9;
static const int kTypeCodeDoubleList = 10;
static const int kTypeCodeTensorList = 11;
static const int kTypeCodeList = 12;

static const int kTypeCodeDictStringKey = 13;
static const int kTypeCodeDictLongKey = 14;

/**
 * PyTorch Lite
 *
 * @author guoweifeng
 * @date 2024/1/10 10:37
 */
@interface TorchModule : NSObject

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath
NS_SWIFT_NAME(init(fileAtPath:))NS_DESIGNATED_INITIALIZER;
+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;
- (void)destroy;
- (NSDictionary *)forwardArray:(NSArray *)array NS_SWIFT_NAME(forward(array:));

@end

NS_ASSUME_NONNULL_END
