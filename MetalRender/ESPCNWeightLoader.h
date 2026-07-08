#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@interface ESPCNWeightLoader : NSObject

@property (nonatomic, weak) id<MTLDevice> device;  // 添加这行

@property (nonatomic, strong) id<MTLBuffer> conv1Weight;
@property (nonatomic, strong) id<MTLBuffer> conv1Bias;
@property (nonatomic, strong) id<MTLBuffer> conv2Weight;
@property (nonatomic, strong) id<MTLBuffer> conv2Bias;
@property (nonatomic, strong) id<MTLBuffer> conv3Weight;
@property (nonatomic, strong) id<MTLBuffer> conv3Bias;

- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (BOOL)loadWeightsFromBundle;
- (BOOL)loadWeightsFromFiles:(NSDictionary *)filePaths;

@end
