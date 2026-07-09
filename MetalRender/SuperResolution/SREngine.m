#import "SREngine.h"
#import "DeepLearningSR.h"

#define YUV_WIDTH  1280
#define YUV_HEIGHT 1280

typedef struct { uint32_t x; uint32_t y; } UInt2;

@interface SREngine ()
@property (nonatomic, strong) id<MTLDevice> device;

@property (nonatomic, strong) id<MTLComputePipelineState> lanczosHorizontalPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> lanczosVerticalPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> downsampleBlurPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> errorComputePipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> upscaleErrorPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> backProjectPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> blendTemporalPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> usmSharpenPipeline;

@property (nonatomic, strong) id<MTLTexture> prevHighResY;
@property (nonatomic, assign) CGSize prevTargetSize;
@property (nonatomic, assign) BOOL temporalHistoryValid;

@property (nonatomic, strong) DeepLearningSR *deepSR;
@end

@implementation SREngine

- (instancetype)initWithDevice:(id<MTLDevice>)device {
    self = [super init];
    if (self) {
        self.device = device;
        self.mode = SRModeTemporalIBP;
        self.ibpLambda = 0.25;
        self.ibpIterations = 4;
        self.temporalWeight = 0.85;
        self.sharpenIntensity = 0.4;
        self.temporalHistoryValid = NO;
        [self setupPipelines];
        _deepSR = [[DeepLearningSR alloc] initWithDevice:device];
    }
    return self;
}

- (void)setupPipelines {
    id<MTLLibrary> lib = [self.device newDefaultLibrary];
    NSArray *names = @[@"lanczos_horizontal", @"lanczos_vertical",
                       @"downsample_blur", @"error_compute",
                       @"upscale_error", @"back_project",
                       @"blend_temporal", @"usm_sharpen"];
    for (NSString *name in names) {
        id<MTLFunction> f = [lib newFunctionWithName:name];
        NSError *err;
        id<MTLComputePipelineState> p = [self.device newComputePipelineStateWithFunction:f error:&err];
        if (!p) { NSLog(@"SREngine: Failed to create %@: %@", name, err); continue; }
        if ([name isEqualToString:@"lanczos_horizontal"]) self.lanczosHorizontalPipeline = p;
        else if ([name isEqualToString:@"lanczos_vertical"]) self.lanczosVerticalPipeline = p;
        else if ([name isEqualToString:@"downsample_blur"]) self.downsampleBlurPipeline = p;
        else if ([name isEqualToString:@"error_compute"]) self.errorComputePipeline = p;
        else if ([name isEqualToString:@"upscale_error"]) self.upscaleErrorPipeline = p;
        else if ([name isEqualToString:@"back_project"]) self.backProjectPipeline = p;
        else if ([name isEqualToString:@"blend_temporal"]) self.blendTemporalPipeline = p;
        else if ([name isEqualToString:@"usm_sharpen"]) self.usmSharpenPipeline = p;
    }
}

- (void)resetHistory {
    self.temporalHistoryValid = NO;
    self.prevHighResY = nil;
}

#pragma mark - Lanczos

- (id<MTLTexture>)upscaleTextureLanczos:(id<MTLTexture>)src
                                toWidth:(NSUInteger)width height:(NSUInteger)height
                          commandBuffer:(id<MTLCommandBuffer>)cb {
    MTLTextureDescriptor *tmpDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                       width:width height:src.height mipmapped:NO];
    tmpDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> tmpTex = [self.device newTextureWithDescriptor:tmpDesc];
    
    MTLTextureDescriptor *outDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                       width:width height:height mipmapped:NO];
    outDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> outTex = [self.device newTextureWithDescriptor:outDesc];
    
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:self.lanczosHorizontalPipeline];
        [enc setTexture:src atIndex:0];
        [enc setTexture:tmpTex atIndex:1];
        struct { uint srcW, srcH, dstW, dstH; } p = {src.width, src.height, tmpTex.width, tmpTex.height};
        [enc setBytes:&p length:sizeof(p) atIndex:0];
        MTLSize grid = MTLSizeMake(tmpTex.width, tmpTex.height, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc endEncoding];
    }
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:self.lanczosVerticalPipeline];
        [enc setTexture:tmpTex atIndex:0];
        [enc setTexture:outTex atIndex:1];
        struct { uint srcW, srcH, dstW, dstH; } p = {tmpTex.width, tmpTex.height, outTex.width, outTex.height};
        [enc setBytes:&p length:sizeof(p) atIndex:0];
        MTLSize grid = MTLSizeMake(outTex.width, outTex.height, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc endEncoding];
    }
    return outTex;
}

- (id<MTLTexture>)newTextureWithWidth:(NSUInteger)width height:(NSUInteger)height {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                     width:width height:height mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    return [self.device newTextureWithDescriptor:desc];
}

#pragma mark - IBP

- (id<MTLTexture>)runIBPIterationsOn:(id<MTLTexture>)highInit
                           reference:(id<MTLTexture>)lowRes
                         commandBuffer:(id<MTLCommandBuffer>)cb {
    id<MTLTexture> highRes = highInit;
    float lambda = self.ibpLambda;
    for (int iter = 0; iter < self.ibpIterations; ++iter) {
        id<MTLTexture> simLow = [self newTextureWithWidth:lowRes.width height:lowRes.height];
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:self.downsampleBlurPipeline];
            [enc setTexture:highRes atIndex:0];
            [enc setTexture:simLow atIndex:1];
            UInt2 srcSize = {highRes.width, highRes.height};
            UInt2 dstSize = {simLow.width, simLow.height};
            [enc setBytes:&srcSize length:sizeof(UInt2) atIndex:0];
            [enc setBytes:&dstSize length:sizeof(UInt2) atIndex:1];
            MTLSize grid = MTLSizeMake(dstSize.x, dstSize.y, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
        }
        
        id<MTLTexture> errorLow = [self newTextureWithWidth:lowRes.width height:lowRes.height];
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:self.errorComputePipeline];
            [enc setTexture:lowRes atIndex:0];
            [enc setTexture:simLow atIndex:1];
            [enc setTexture:errorLow atIndex:2];
            MTLSize grid = MTLSizeMake(lowRes.width, lowRes.height, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
        }
        
        id<MTLTexture> errorHigh = [self newTextureWithWidth:highRes.width height:highRes.height];
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:self.upscaleErrorPipeline];
            [enc setTexture:errorLow atIndex:0];
            [enc setTexture:errorHigh atIndex:1];
            UInt2 highSize = {highRes.width, highRes.height};
            [enc setBytes:&highSize length:sizeof(UInt2) atIndex:0];
            MTLSize grid = MTLSizeMake(highSize.x, highSize.y, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
        }
        
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:self.backProjectPipeline];
            [enc setTexture:highRes atIndex:0];
            [enc setTexture:errorHigh atIndex:1];
            [enc setBytes:&lambda length:sizeof(float) atIndex:0];
            MTLSize grid = MTLSizeMake(highRes.width, highRes.height, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
        }
        lambda *= 0.9;
    }
    return highRes;
}

- (id<MTLTexture>)applyTemporalIBP:(id<MTLTexture>)yLow
                       targetSize:(CGSize)targetSize
                    commandBuffer:(id<MTLCommandBuffer>)cb {
    id<MTLTexture> curLanczos = [self upscaleTextureLanczos:yLow
                                                    toWidth:targetSize.width height:targetSize.height
                                              commandBuffer:cb];
    
    id<MTLTexture> initEstimate = curLanczos;
    if (self.temporalHistoryValid && self.prevHighResY &&
        self.prevHighResY.width == targetSize.width &&
        self.prevHighResY.height == targetSize.height) {
        id<MTLTexture> blended = [self newTextureWithWidth:targetSize.width height:targetSize.height];
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:self.blendTemporalPipeline];
            [enc setTexture:curLanczos atIndex:0];
            [enc setTexture:self.prevHighResY atIndex:1];
            [enc setTexture:blended atIndex:2];
            float weight = self.temporalWeight;
            [enc setBytes:&weight length:sizeof(float) atIndex:0];
            MTLSize grid = MTLSizeMake(targetSize.width, targetSize.height, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
        }
        initEstimate = blended;
    }
    
    id<MTLTexture> afterIBP = [self runIBPIterationsOn:initEstimate reference:yLow commandBuffer:cb];
    
    if (self.temporalHistoryValid && self.prevHighResY) {
        id<MTLTexture> finalResult = [self newTextureWithWidth:targetSize.width height:targetSize.height];
        float alpha = 0.2;
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:self.blendTemporalPipeline];
            [enc setTexture:afterIBP atIndex:0];
            [enc setTexture:self.prevHighResY atIndex:1];
            [enc setTexture:finalResult atIndex:2];
            float historyWeight = 1.0 - alpha;
            [enc setBytes:&historyWeight length:sizeof(float) atIndex:0];
            MTLSize grid = MTLSizeMake(targetSize.width, targetSize.height, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
        }
        afterIBP = finalResult;
    }
    
    id<MTLTexture> historyCopy = [self newTextureWithWidth:targetSize.width height:targetSize.height];
    {
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromTexture:afterIBP toTexture:historyCopy];
        [blit endEncoding];
    }
    self.prevHighResY = historyCopy;
    self.temporalHistoryValid = YES;
    self.prevTargetSize = targetSize;
    
    return afterIBP;
}

- (id<MTLTexture>)sharpenTexture:(id<MTLTexture>)src
                   commandBuffer:(id<MTLCommandBuffer>)cb {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:src.pixelFormat
                                                                                     width:src.width height:src.height mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> outTex = [self.device newTextureWithDescriptor:desc];
    
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:self.usmSharpenPipeline];
    [enc setTexture:src atIndex:0];
    [enc setTexture:outTex atIndex:1];
    [enc setBytes:&_sharpenIntensity length:sizeof(float) atIndex:0];
    MTLSize grid = MTLSizeMake(outTex.width, outTex.height, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [enc endEncoding];
    return outTex;
}

#pragma mark - 主入口

- (id<MTLTexture>)processYTexture:(id<MTLTexture>)yLow
                  uTexture:(id<MTLTexture>)uLow
                  vTexture:(id<MTLTexture>)vLow
                targetSize:(CGSize)targetSize
             commandBuffer:(id<MTLCommandBuffer>)cb
         outUTexture:(id<MTLTexture> * _Nullable)outU
         outVTexture:(id<MTLTexture> * _Nullable)outV
{
    id<MTLTexture> yOut = nil;
    id<MTLTexture> uOut = nil;
    id<MTLTexture> vOut = nil;
    
    if (self.mode != SRModeOff) {
        uOut = [self upscaleTextureLanczos:uLow toWidth:targetSize.width/2 height:targetSize.height/2 commandBuffer:cb];
        vOut = [self upscaleTextureLanczos:vLow toWidth:targetSize.width/2 height:targetSize.height/2 commandBuffer:cb];
    } else {
        uOut = uLow;
        vOut = vLow;
    }
    
    switch (self.mode) {
        case SRModeOff:
            yOut = yLow;
            break;
        case SRModeLanczos:
            yOut = [self upscaleTextureLanczos:yLow toWidth:targetSize.width height:targetSize.height commandBuffer:cb];
            break;
        case SRModeIBP: {
            id<MTLTexture> init = [self upscaleTextureLanczos:yLow toWidth:targetSize.width height:targetSize.height commandBuffer:cb];
            yOut = [self runIBPIterationsOn:init reference:yLow commandBuffer:cb];
            self.temporalHistoryValid = NO;
            break;
        }
        case SRModeTemporalIBP:
            yOut = [self applyTemporalIBP:yLow targetSize:targetSize commandBuffer:cb];
            break;
        case SRModeDeepLearning:
            yOut = [self.deepSR processYTexture:yLow targetSize:targetSize commandBuffer:cb];
            break;
    }
    
    if (self.mode != SRModeOff && yOut) {
        yOut = [self sharpenTexture:yOut commandBuffer:cb];
    }
    
    if (outU) *outU = uOut;
    if (outV) *outV = vOut;
    return yOut;
}

@end
