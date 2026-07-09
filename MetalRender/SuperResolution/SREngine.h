#import <Metal/Metal.h>

typedef NS_ENUM(NSInteger, SRMode) {
    SRModeOff = 0,
    SRModeLanczos,
    SRModeIBP,
    SRModeTemporalIBP,
    SRModeDeepLearning
};

@interface SREngine : NSObject

@property (nonatomic, assign) SRMode mode;
@property (nonatomic, assign) float ibpLambda;
@property (nonatomic, assign) int ibpIterations;
@property (nonatomic, assign) float temporalWeight;
@property (nonatomic, assign) float sharpenIntensity;

- (instancetype)initWithDevice:(id<MTLDevice>)device;

- (id<MTLTexture>)processYTexture:(id<MTLTexture>)yLow
                  uTexture:(id<MTLTexture>)uLow
                  vTexture:(id<MTLTexture>)vLow
                targetSize:(CGSize)targetSize
             commandBuffer:(id<MTLCommandBuffer>)cb
         outUTexture:(id<MTLTexture> * _Nullable)outU
         outVTexture:(id<MTLTexture> * _Nullable)outV;

- (void)resetHistory;

@end
