#import "DeepLearningSR.h"
#import <Metal/Metal.h>

// 网络结构
#define SRCNN_IN_CH      1
#define SRCNN_HIDDEN     64    // 第一层输出通道数
#define SRCNN_HIDDEN2    32    // 第二层输出通道数
#define SRCNN_OUT_CH     1     // 输出残差（单通道）

// 性能保护：如果目标尺寸超过此限制，先将输入缩小至限制再处理
#define SRCNN_MAX_TARGET_WIDTH  1920

typedef struct { uint inCh, outCh, width, height; } ConvDims;

@interface DeepLearningSR ()
@property (nonatomic, strong) id<MTLDevice> device;

// 管线
@property (nonatomic, strong) id<MTLComputePipelineState> convPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> bilinearPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> add2DPipeline;

// 权重缓冲区（三层卷积）
@property (nonatomic, strong) id<MTLBuffer> w1, b1;   // conv1: 1 -> 64
@property (nonatomic, strong) id<MTLBuffer> w2, b2;   // conv2: 64 -> 32
@property (nonatomic, strong) id<MTLBuffer> w3, b3;   // conv3: 32 -> 1

// 特征图纹理缓存（二维数组）
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
    self.convPipeline     = [self.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"conv3x3_relu"] error:nil];
    self.bilinearPipeline = [self.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bilinear_upsample"] error:nil];
    self.add2DPipeline    = [self.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"add_2d"] error:nil];
}

- (void)loadWeights {
    // 占位权重：全部设为极小值，使残差接近0 → 输出≈双线性上采样
    // conv1: 1->64, 3x3
    NSUInteger w1Size = SRCNN_HIDDEN * SRCNN_IN_CH * 9;
    float *w1 = (float*)calloc(w1Size, sizeof(float));      // 全零
    self.w1 = [self.device newBufferWithBytes:w1 length:w1Size*sizeof(float) options:MTLResourceStorageModeShared];
    free(w1);
    float *b1 = (float*)calloc(SRCNN_HIDDEN, sizeof(float));
    self.b1 = [self.device newBufferWithBytes:b1 length:SRCNN_HIDDEN*sizeof(float) options:MTLResourceStorageModeShared];
    free(b1);

    // conv2: 64->32
    NSUInteger w2Size = SRCNN_HIDDEN2 * SRCNN_HIDDEN * 9;
    float *w2 = (float*)calloc(w2Size, sizeof(float));
    self.w2 = [self.device newBufferWithBytes:w2 length:w2Size*sizeof(float) options:MTLResourceStorageModeShared];
    free(w2);
    float *b2 = (float*)calloc(SRCNN_HIDDEN2, sizeof(float));
    self.b2 = [self.device newBufferWithBytes:b2 length:SRCNN_HIDDEN2*sizeof(float) options:MTLResourceStorageModeShared];
    free(b2);

    // conv3: 32->1
    NSUInteger w3Size = SRCNN_OUT_CH * SRCNN_HIDDEN2 * 9;
    float *w3 = (float*)calloc(w3Size, sizeof(float));
    self.w3 = [self.device newBufferWithBytes:w3 length:w3Size*sizeof(float) options:MTLResourceStorageModeShared];
    free(w3);
    float *b3 = (float*)calloc(SRCNN_OUT_CH, sizeof(float));
    self.b3 = [self.device newBufferWithBytes:b3 length:SRCNN_OUT_CH*sizeof(float) options:MTLResourceStorageModeShared];
    free(b3);
}

#pragma mark - 纹理缓存

- (id<MTLTexture>)texture2DWithWidth:(NSUInteger)w height:(NSUInteger)h pixelFormat:(MTLPixelFormat)fmt {
    NSString *key = [NSString stringWithFormat:@"2D_%lu_%lu_%lu", w, h, (unsigned long)fmt];
    id<MTLTexture> tex = self.textureCache[key];
    if (!tex) {
        MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt
                                                                                         width:w height:h mipmapped:NO];
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        tex = [self.device newTextureWithDescriptor:desc];
        self.textureCache[key] = tex;
    }
    return tex;
}

- (id<MTLTexture>)textureArrayWithWidth:(NSUInteger)w height:(NSUInteger)h channels:(NSUInteger)c {
    NSString *key = [NSString stringWithFormat:@"array_%lu_%lu_%lu", w, h, c];
    id<MTLTexture> tex = self.textureCache[key];
    if (!tex) {
        MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                         width:w height:h mipmapped:NO];
        desc.textureType = MTLTextureType2DArray;
        desc.arrayLength = c;
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        tex = [self.device newTextureWithDescriptor:desc];
        self.textureCache[key] = tex;
    }
    return tex;
}

#pragma mark - 工具函数

- (void)runConv:(id<MTLComputeCommandEncoder>)enc
      input:(id<MTLTexture>)inTex output:(id<MTLTexture>)outTex
    weights:(id<MTLBuffer>)w bias:(id<MTLBuffer>)b
    inCh:(uint)inCh outCh:(uint)outCh {
    [enc setComputePipelineState:self.convPipeline];
    [enc setTexture:inTex atIndex:0];
    [enc setTexture:outTex atIndex:1];
    [enc setBuffer:w offset:0 atIndex:0];
    [enc setBuffer:b offset:0 atIndex:1];
    ConvDims d = {inCh, outCh, (uint)outTex.width, (uint)outTex.height};
    [enc setBytes:&d length:sizeof(ConvDims) atIndex:2];
    MTLSize grid = MTLSizeMake(outTex.width, outTex.height, outCh);
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
}

- (id<MTLTexture>)bilinearUpsample:(id<MTLTexture>)src
                          toWidth:(NSUInteger)w height:(NSUInteger)h
                    commandBuffer:(id<MTLCommandBuffer>)cb {
    id<MTLTexture> dst = [self texture2DWithWidth:w height:h pixelFormat:src.pixelFormat];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:self.bilinearPipeline];
    [enc setTexture:src atIndex:0];
    [enc setTexture:dst atIndex:1];
    MTLSize grid = MTLSizeMake(w, h, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [enc endEncoding];
    return dst;
}

- (id<MTLTexture>)addTextures:(id<MTLTexture>)texA second:(id<MTLTexture>)texB
                commandBuffer:(id<MTLCommandBuffer>)cb {
    id<MTLTexture> out = [self texture2DWithWidth:texA.width height:texA.height pixelFormat:texA.pixelFormat];
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

#pragma mark - SRCNN 主流程

- (nullable id<MTLTexture>)processYTexture:(id<MTLTexture>)yLow
                                targetSize:(CGSize)targetSize
                             commandBuffer:(id<MTLCommandBuffer>)cb {
    NSUInteger finalW = (NSUInteger)targetSize.width;
    NSUInteger finalH = (NSUInteger)targetSize.height;

    // 1. 双线性上采样到目标尺寸
    id<MTLTexture> base = [self bilinearUpsample:yLow toWidth:finalW height:finalH commandBuffer:cb];

    // 2. 如果目标尺寸过大，先缩小再处理以提升速度（SRCNN 对尺寸敏感）
    NSUInteger workW = finalW, workH = finalH;
    if (workW > SRCNN_MAX_TARGET_WIDTH || workH > SRCNN_MAX_TARGET_WIDTH) {
        float scale = (float)SRCNN_MAX_TARGET_WIDTH / MAX(workW, workH);
        workW = (NSUInteger)(workW * scale);
        workH = (NSUInteger)(workH * scale);
        // 缩小 base 到工作尺寸
        base = [self bilinearUpsample:base toWidth:workW height:workH commandBuffer:cb]; // 实际上这里应该用 downsample，但 bilinear 缩小也接受（采样器可实现），这里简化用 bilinear，效果正确
    }

    // 3. 三层卷积（在同一个 encoder 中完成）
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    // Conv1: 1 → 64
    id<MTLTexture> feat1 = [self textureArrayWithWidth:workW height:workH channels:SRCNN_HIDDEN];
    [self runConv:enc input:base output:feat1 weights:_w1 bias:_b1 inCh:1 outCh:SRCNN_HIDDEN];

    // Conv2: 64 → 32
    id<MTLTexture> feat2 = [self textureArrayWithWidth:workW height:workH channels:SRCNN_HIDDEN2];
    [self runConv:enc input:feat1 output:feat2 weights:_w2 bias:_b2 inCh:SRCNN_HIDDEN outCh:SRCNN_HIDDEN2];

    // Conv3: 32 → 1
    id<MTLTexture> residual = [self texture2DWithWidth:workW height:workH pixelFormat:MTLPixelFormatR32Float];
    [self runConv:enc input:feat2 output:residual weights:_w3 bias:_b3 inCh:SRCNN_HIDDEN2 outCh:1];
    [enc endEncoding];

    // 4. 残差相加（注意像素格式）
    // base 是 R8Unorm，residual 是 R32Float，需要统一为 R32Float 相加，再转回 R8Unorm
    // 简单做法：将 base 转为 R32Float，相加，再转为 R8Unorm（利用 add_2d 自动转换）
    // 但 add_2d 要求两个纹理格式相同，这里 base 是 R8Unorm，residual 是 R32Float，不能直接相加。
    // 我们先将 base 转换为 R32Float（通过 bilinear 缩放，实际上用 bilinear 采样可以转换格式）
    id<MTLTexture> baseFloat = [self texture2DWithWidth:workW height:workH pixelFormat:MTLPixelFormatR32Float];
    // 用一个简单的复制/转换 kernel，但这里可以直接用 bilinear 采样从 R8Unorm 到 R32Float（Metal 自动转换）
    // 我们写一个新的 encoder 使用 bilinear_upsample 从 base(R8) -> baseFloat(R32)，尺寸相同，会触发格式转换
    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:self.bilinearPipeline];
    [enc setTexture:base atIndex:0];
    [enc setTexture:baseFloat atIndex:1];
    MTLSize grid = MTLSizeMake(workW, workH, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [enc endEncoding];

    // 相加 (R32Float + R32Float)
    id<MTLTexture> resultFloat = [self texture2DWithWidth:workW height:workH pixelFormat:MTLPixelFormatR32Float];
    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:self.add2DPipeline];
    [enc setTexture:baseFloat atIndex:0];
    [enc setTexture:residual atIndex:1];
    [enc setTexture:resultFloat atIndex:2];
    grid = MTLSizeMake(workW, workH, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [enc endEncoding];

    // 转换回 R8Unorm（最终输出）
    id<MTLTexture> result = [self texture2DWithWidth:workW height:workH pixelFormat:MTLPixelFormatR8Unorm];
    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:self.bilinearPipeline]; // 利用 bilinear 采样转换格式
    [enc setTexture:resultFloat atIndex:0];
    [enc setTexture:result atIndex:1];
    grid = MTLSizeMake(workW, workH, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [enc endEncoding];

    // 5. 如果之前缩小过，则放大回最终尺寸
    if (workW != finalW || workH != finalH) {
        result = [self bilinearUpsample:result toWidth:finalW height:finalH commandBuffer:cb];
    }

    return result;
}

@end
