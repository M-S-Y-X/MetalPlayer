#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

@interface DeepLearningSR : NSObject

- (instancetype)initWithDevice:(id<MTLDevice>)device;

- (nullable id<MTLTexture>)processYTexture:(id<MTLTexture>)yLow
                                targetSize:(CGSize)targetSize
                             commandBuffer:(id<MTLCommandBuffer>)commandBuffer;

@end

NS_ASSUME_NONNULL_END
