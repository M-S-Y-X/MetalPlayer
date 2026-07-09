#import "DeepLearningSR.h"
#import <Metal/Metal.h>
#import <math.h>
#import <stdlib.h>

// 与你的权重文件匹配的通道数
#define SRCNN_IN_CH      1
#define SRCNN_HIDDEN     128    // 第一层输出
#define SRCNN_HIDDEN2    64     // 第二层输出
#define SRCNN_OUT_CH     1

// 每层的卷积核大小
#define SRCNN_K1         9
#define SRCNN_K2         3
#define SRCNN_K3         5

// 性能：最大处理宽度
#define SRCNN_MAX_TARGET_WIDTH  1920

typedef struct { uint inCh, outCh, width, height; } ConvDims;

@interface DeepLearningSR ()
@property (nonatomic, strong) id<MTLDevice> device;

@property (nonatomic, strong) id<MTLComputePipelineState> convPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> bilinearPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> add2DPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> convertPipeline;

@property (nonatomic, strong) id<MTLBuffer> w1, b1, w2, b2, w3, b3;
@property (nonatomic, strong) NSMutableDictionary<NSString*, id<MTLTexture>> *textureCache;
@end

@implementation DeepLearningSR

- (instancetype)initWithDevice:(id<MTLDevice>)device {
    self = [super init];
    if (self) {
        self.device = device;
        _textureCache = [NSMutableDictionary dictionary];
        [self setupPipelines];
        [self loadWeights];
    }
    return self;
}

- (void)setupPipelines {
    id<MTLLibrary> lib = [self.device newDefaultLibrary];
    self.convPipeline      = [self.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"conv2d_relu"] error:nil];
    self.bilinearPipeline  = [self.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bilinear_upsample"] error:nil];
    self.add2DPipeline     = [self.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"add_2d"] error:nil];
    self.convertPipeline   = [self.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"convert_to_r8unorm"] error:nil];
}

- (void)loadWeights {
    // 从 Bundle 加载 .bin 文件，失败则回退到随机权重
    id<MTLBuffer> (^load)(NSString*) = ^id<MTLBuffer>(NSString *name) {
        NSString *path = [[NSBundle mainBundle] pathForResource:name ofType:@"bin"];
        if (!path) return nil;
        NSData *data = [NSData dataWithContentsOfFile:path];
        return data.length ? [self.device newBufferWithBytes:data.bytes length:data.length options:MTLResourceStorageModeShared] : nil;
    };

    self.w1 = load(@"conv1_weight"); self.b1 = load(@"conv1_bias");
    self.w2 = load(@"conv2_weight"); self.b2 = load(@"conv2_bias");
    self.w3 = load(@"conv3_weight"); self.b3 = load(@"conv3_bias");

    if (!self.w1 || !self.b1 || !self.w2 || !self.b2 || !self.w3 || !self.b3) {
        NSLog(@"⚠️ 真实权重未找到，使用随机权重（效果等同双线性放大）");
        [self loadRandomWeights];
    }
}

- (void)loadRandomWeights {
    id<MTLBuffer> (^fill)(NSUInteger) = ^id<MTLBuffer>(NSUInteger count) {
        float *d = malloc(count*sizeof(float));
        for (NSUInteger i=0; i<count; ++i) d[i] = ((float)arc4random()/UINT32_MAX * 0.002f);
        id<MTLBuffer> b = [self.device newBufferWithBytes:d length:count*sizeof(float) options:MTLResourceStorageModeShared];
        free(d); return b;
    };
    self.w1 = fill(SRCNN_HIDDEN * SRCNN_IN_CH * SRCNN_K1*SRCNN_K1);
    self.b1 = fill(SRCNN_HIDDEN);
    self.w2 = fill(SRCNN_HIDDEN2 * SRCNN_HIDDEN * SRCNN_K2*SRCNN_K2);
    self.b2 = fill(SRCNN_HIDDEN2);
    self.w3 = fill(SRCNN_OUT_CH * SRCNN_HIDDEN2 * SRCNN_K3*SRCNN_K3);
    self.b3 = fill(SRCNN_OUT_CH);
}

#pragma mark - 纹理创建（带缓存）

- (id<MTLTexture>)newTexture2DWithWidth:(NSUInteger)w height:(NSUInteger)h pixelFormat:(MTLPixelFormat)fmt {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt width:w height:h mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    return [self.device newTextureWithDescriptor:desc];
}

- (id<MTLTexture>)newTextureArrayWithWidth:(NSUInteger)w height:(NSUInteger)h channels:(NSUInteger)c {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                     width:w height:h mipmapped:NO];
    desc.textureType = MTLTextureType2DArray;
    desc.arrayLength = c;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    return [self.device newTextureWithDescriptor:desc];
}

#pragma mark - 卷积 / 采样 / 加法

- (void)runConv:(id<MTLComputeCommandEncoder>)enc
      input:(id<MTLTexture>)inTex output:(id<MTLTexture>)outTex
    weights:(id<MTLBuffer>)w bias:(id<MTLBuffer>)b
    inCh:(uint)inCh outCh:(uint)outCh kernelSize:(uint)kSize {
    [enc setComputePipelineState:self.convPipeline];
    [enc setTexture:inTex atIndex:0];
    [enc setTexture:outTex atIndex:1];
    [enc setBuffer:w offset:0 atIndex:0];
    [enc setBuffer:b offset:0 atIndex:1];
    ConvDims d = {inCh, outCh, (uint)outTex.width, (uint)outTex.height};
    [enc setBytes:&d length:sizeof(ConvDims) atIndex:2];
    [enc setBytes:&kSize length:sizeof(uint) atIndex:3];
    MTLSize grid = MTLSizeMake(outTex.width, outTex.height, outCh);
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
}

- (id<MTLTexture>)bilinearUpsample:(id<MTLTexture>)src toWidth:(NSUInteger)w height:(NSUInteger)h
                    commandBuffer:(id<MTLCommandBuffer>)cb outputFormat:(MTLPixelFormat)outputFormat {
    id<MTLTexture> dst = [self newTexture2DWithWidth:w height:h pixelFormat:outputFormat];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:self.bilinearPipeline];
    [enc setTexture:src atIndex:0];
    [enc setTexture:dst atIndex:1];
    MTLSize grid = MTLSizeMake(w, h, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [enc endEncoding];
    return dst;
}

- (id<MTLTexture>)addTextures:(id<MTLTexture>)texA second:(id<MTLTexture>)texB commandBuffer:(id<MTLCommandBuffer>)cb {
    id<MTLTexture> out = [self newTexture2DWithWidth:texA.width height:texA.height pixelFormat:MTLPixelFormatR32Float];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:self.add2DPipeline];
    [enc setTexture:texA atIndex:0];
    [enc setTexture:texB atIndex:1];
    [enc setTexture:out atIndex:2];
    MTLSize grid = MTLSizeMake(texA.width, texA.height, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [enc endEncoding];
    return out;
}

- (id<MTLTexture>)convertToR8:(id<MTLTexture>)src commandBuffer:(id<MTLCommandBuffer>)cb {
    id<MTLTexture> dst = [self newTexture2DWithWidth:src.width height:src.height pixelFormat:MTLPixelFormatR8Unorm];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:self.convertPipeline];
    [enc setTexture:src atIndex:0];
    [enc setTexture:dst atIndex:1];
    MTLSize grid = MTLSizeMake(src.width, src.height, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [enc endEncoding];
    return dst;
}

#pragma mark - 推理

- (nullable id<MTLTexture>)processYTexture:(id<MTLTexture>)yLow
                                targetSize:(CGSize)targetSize
                             commandBuffer:(id<MTLCommandBuffer>)cb {
    NSUInteger finalW = (NSUInteger)targetSize.width;
    NSUInteger finalH = (NSUInteger)targetSize.height;

    // 1. 双线性上采样到目标尺寸 (R32Float)
    id<MTLTexture> base = [self bilinearUpsample:yLow toWidth:finalW height:finalH commandBuffer:cb outputFormat:MTLPixelFormatR32Float];

    // 2. 如果尺寸过大，缩小到工作尺寸
    NSUInteger workW = finalW, workH = finalH;
    id<MTLTexture> workBase = base;
    if (workW > SRCNN_MAX_TARGET_WIDTH || workH > SRCNN_MAX_TARGET_WIDTH) {
        float scale = (float)SRCNN_MAX_TARGET_WIDTH / MAX(workW, workH);
        workW = (NSUInteger)(workW * scale);
        workH = (NSUInteger)(workH * scale);
        workBase = [self bilinearUpsample:base toWidth:workW height:workH commandBuffer:cb outputFormat:MTLPixelFormatR32Float];
    }

    // 3. 三层卷积 (在同一个 encoder 中完成以提高效率)
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    // Conv1: 1 → 128, 9x9
    id<MTLTexture> feat1 = [self newTextureArrayWithWidth:workW height:workH channels:SRCNN_HIDDEN];
    [self runConv:enc input:workBase output:feat1 weights:_w1 bias:_b1 inCh:1 outCh:SRCNN_HIDDEN kernelSize:SRCNN_K1];

    // Conv2: 128 → 64, 3x3
    id<MTLTexture> feat2 = [self newTextureArrayWithWidth:workW height:workH channels:SRCNN_HIDDEN2];
    [self runConv:enc input:feat1 output:feat2 weights:_w2 bias:_b2 inCh:SRCNN_HIDDEN outCh:SRCNN_HIDDEN2 kernelSize:SRCNN_K2];

    // Conv3: 64 → 1, 5x5
    id<MTLTexture> residual = [self newTexture2DWithWidth:workW height:workH pixelFormat:MTLPixelFormatR32Float];
    [self runConv:enc input:feat2 output:residual weights:_w3 bias:_b3 inCh:SRCNN_HIDDEN2 outCh:1 kernelSize:SRCNN_K3];
    [enc endEncoding];

    // 4. 残差相加
    id<MTLTexture> resultFloat = [self addTextures:workBase second:residual commandBuffer:cb];

    // 5. 转为 R8Unorm
    id<MTLTexture> result = [self convertToR8:resultFloat commandBuffer:cb];

    // 6. 如果缩小过，再放大回最终尺寸
    if (workW != finalW || workH != finalH) {
        result = [self bilinearUpsample:result toWidth:finalW height:finalH commandBuffer:cb outputFormat:MTLPixelFormatR8Unorm];
    }

    return result;
}

@end
