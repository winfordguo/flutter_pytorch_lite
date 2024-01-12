#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

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
- (NSDictionary *)forwardMap:(NSDictionary *)map NS_SWIFT_NAME(forward(map:));

@end

NS_ASSUME_NONNULL_END
