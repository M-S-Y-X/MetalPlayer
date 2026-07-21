//
//  ImageStitch.m
//  分步调试版：返回不同阶段的纹理，便于定位
//

#import "ImageStitch.h"
#import "StitchTypes.h"
#import <simd/simd.h>

// 调试模式：注释掉下面宏定义以切换返回值
// #define DEBUG_RETURN_WARPB      // 返回 warpB
// #define DEBUG_RETURN_CANVASA    // 返回 canvasA
// #define DEBUG_RETURN_MASK       // 返回 maskA 或 maskB
// #define DEBUG_RETURN_WEIGHT     // 返回 weight

// 当前可用的宏（取消注释你想要测试的阶段）
// #define DEBUG_RETURN_WARPB
// #define DEBUG_RETURN_CANVASA
// #define DEBUG_RETURN_MASK
// #define DEBUG_RETURN_WEIGHT

@interface ImageStitcher()
{
    id<MTLDevice> _device;
    id<MTLLibrary> _library;
    
    id<MTLComputePipelineState> _warpPipeline;
    id<MTLComputePipelineState> _maskPipeline;
    id<MTLComputePipelineState> _distanceInitPipeline;
    id<MTLComputePipelineState> _distanceJFAPipeline;
    id<MTLComputePipelineState> _weightPipeline;
    id<MTLComputePipelineState> _blendPipeline;
    id<MTLComputePipelineState> _downPipeline;
    id<MTLComputePipelineState> _upPipeline;
    id<MTLComputePipelineState> _lapPipeline;
    id<MTLComputePipelineState> _reconstructPipeline;
    NSUInteger _outputWidth;
    NSUInteger _outputHeight;
}
@end

@implementation ImageStitcher

- (void)debugTexture:(id<MTLTexture>)texture
                name:(NSString *)name
{
    if(texture == nil)
    {
        NSLog(@"%@ texture nil",name);
        return;
    }


    uint8_t pixel[16]={0};


    NSUInteger x = texture.width/2;
    NSUInteger y = texture.height/2;


    MTLRegion region =
    MTLRegionMake2D(x,y,1,1);


    [texture getBytes:pixel
          bytesPerRow:4*4
           fromRegion:region
          mipmapLevel:0];


    NSLog(@"%@ pixel RGBA=(%d,%d,%d,%d)",
          name,
          pixel[0],
          pixel[1],
          pixel[2],
          pixel[3]);
}

- (instancetype)initWithDevice:(id<MTLDevice>)device
{
    self = [super init];
    if (self) {
        _device = device;
        [self setupMetal];
    }
    return self;
}

- (void)setupMetal
{
    _library = [_device newDefaultLibrary];
    _warpPipeline = [self pipeline:@"warpPerspectiveKernel"];
    _maskPipeline = [self pipeline:@"generateMaskKernel"];
    _distanceInitPipeline = [self pipeline:@"distanceInitKernel"];
    _distanceJFAPipeline = [self pipeline:@"distanceJFAKernel"];
    _weightPipeline = [self pipeline:@"weightKernel"];
    _blendPipeline = [self pipeline:@"pyramidBlendKernel"];
    _downPipeline = [self pipeline:@"gaussianDownKernel"];
    _upPipeline = [self pipeline:@"gaussianUpKernel"];
    _lapPipeline = [self pipeline:@"laplacianKernel"];
    _reconstructPipeline = [self pipeline:@"reconstructKernel"];
}

- (id<MTLComputePipelineState>)pipeline:(NSString*)name
{
    id<MTLFunction> fn = [_library newFunctionWithName:name];
    if (!fn) {
        NSLog(@"Shader missing: %@", name);
        return nil;
    }
    NSError *error = nil;
    id<MTLComputePipelineState> state = [_device newComputePipelineStateWithFunction:fn error:&error];
    if (error) {
        NSLog(@"Pipeline error %@ : %@", name, error);
    }
    return state;
}

- (id<MTLTexture>)createTexture:(NSUInteger)width
                         height:(NSUInteger)height
                         format:(MTLPixelFormat)format
{
    if(width==0 || height==0)
        return nil;

    MTLTextureDescriptor *desc =
    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                       width:width
                                                      height:height
                                                   mipmapped:NO];

    desc.usage =
        MTLTextureUsageShaderRead |
        MTLTextureUsageShaderWrite;

    return [_device newTextureWithDescriptor:desc];
}

- (void)dispatch:(id<MTLComputePipelineState>)pipeline encoder:(id<MTLComputeCommandEncoder>)encoder width:(NSUInteger)width height:(NSUInteger)height
{
    if (!pipeline) return;
    MTLSize grid = MTLSizeMake(width, height, 1);
    NSUInteger tw = pipeline.threadExecutionWidth;
    NSUInteger th = pipeline.maxTotalThreadsPerThreadgroup / tw;
    MTLSize group = MTLSizeMake(tw, th, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    NSLog(@"Warp kernel dispatched");
}

- (void)copyTexture:(id<MTLTexture>)src toTexture:(id<MTLTexture>)dst offsetX:(int)offsetX offsetY:(int)offsetY commandBuffer:(id<MTLCommandBuffer>)commandBuffer
{
    if(src.pixelFormat != dst.pixelFormat)
    {
        NSLog(@"⚠️ format convert needed");
    }
    
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    int dstX = -offsetX;
    int dstY = -offsetY;
    NSUInteger copyW = MIN(src.width, dst.width - dstX);
    NSUInteger copyH = MIN(src.height, dst.height - dstY);
    if (copyW > 0 && copyH > 0) {
        [blit copyFromTexture:src
                   sourceSlice:0
                   sourceLevel:0
                  sourceOrigin:MTLOriginMake(0, 0, 0)
                    sourceSize:MTLSizeMake(copyW, copyH, 1)
                     toTexture:dst
              destinationSlice:0
              destinationLevel:0
             destinationOrigin:MTLOriginMake(dstX,dstY,0)];
    }
    [blit endEncoding];
}

- (id<MTLTexture>)stitchTextureA:(id<MTLTexture>)textureA textureB:(id<MTLTexture>)textureB homography:(simd_float3x3)H canvasWidth:(NSUInteger)canvasWidth canvasHeight:(NSUInteger)canvasHeight offsetX:(int)offsetX offsetY:(int)offsetY commandBuffer:(id<MTLCommandBuffer>)commandBuffer
{
    NSLog(@"========== Stitch Begin ==========");

    NSLog(@"TextureA size=%lux%lu format=%ld",
          textureA.width,
          textureA.height,
          textureA.pixelFormat);

    NSLog(@"TextureB size=%lux%lu format=%ld",
          textureB.width,
          textureB.height,
          textureB.pixelFormat);

    NSLog(@"Canvas=%lux%lu offset=(%d,%d)",
          canvasWidth,
          canvasHeight,
          offsetX,
          offsetY);
    
    _outputWidth = canvasWidth;
    _outputHeight = canvasHeight;
    
    NSLog(@"📐 stitchTextureA: canvas %lu x %lu, offset (%d, %d)", canvasWidth, canvasHeight, offsetX, offsetY);
    
    // 1. Warp B
    id<MTLTexture> warpB = [self createTexture:canvasWidth
                                        height:canvasHeight
                                        format:MTLPixelFormatRGBA8Unorm];
    NSLog(@"WarpB created %lux%lu format=%ld",
          warpB.width,
          warpB.height,
          (long)warpB.pixelFormat);
    
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_warpPipeline];
    [encoder setTexture:textureB atIndex:0];
    [encoder setTexture:warpB atIndex:1];
    
    simd_float3x3 H_inv = simd_inverse(H);
    NSLog(@"H=");
    NSLog(@"[%f %f %f]",
          H.columns[0].x,
          H.columns[1].x,
          H.columns[2].x);

    NSLog(@"[%f %f %f]",
          H.columns[0].y,
          H.columns[1].y,
          H.columns[2].y);

    NSLog(@"[%f %f %f]",
          H.columns[0].z,
          H.columns[1].z,
          H.columns[2].z);


    NSLog(@"H_inv=");
    NSLog(@"[%f %f %f]",
          H_inv.columns[0].x,
          H_inv.columns[1].x,
          H_inv.columns[2].x);

    NSLog(@"[%f %f %f]",
          H_inv.columns[0].y,
          H_inv.columns[1].y,
          H_inv.columns[2].y);

    NSLog(@"[%f %f %f]",
          H_inv.columns[0].z,
          H_inv.columns[1].z,
          H_inv.columns[2].z);
    
    simd_float3 p;

    p.x = canvasWidth *0.5 - offsetX;
    p.y = canvasHeight*0.5 - offsetY;
    p.z = 1.0;


    simd_float3 src =
    simd_mul(H_inv, p);


    src.x /= src.z;
    src.y /= src.z;
    src.z = 1.0;


    NSLog(@"CPU Warp center dst=(%f,%f) src=(%f,%f)",
          p.x,
          p.y,
          src.x,
          src.y);
    
    WarpParams warp;

    warp.H_inv = H_inv;

    warp.width  = (uint32_t)canvasWidth;
    warp.height = (uint32_t)canvasHeight;

    warp.offsetX = offsetX;
    warp.offsetY = offsetY;
    
    [encoder setBytes:&warp length:sizeof(warp) atIndex:0];
    [self dispatch:_warpPipeline encoder:encoder width:canvasWidth height:canvasHeight];
    
    [encoder endEncoding];
    
#ifdef DEBUG_RETURN_WARPB
    NSLog(@"🔍 DEBUG: Returning warpB");
    return warpB;
#endif
    
    // 2. 创建画布 A
    id<MTLTexture> canvasA = [self createTexture:canvasWidth
                                          height:canvasHeight
                                          format:MTLPixelFormatRGBA8Unorm];

    [self copyTexture:textureA toTexture:canvasA offsetX:offsetX offsetY:offsetY commandBuffer:commandBuffer];
    
#ifdef DEBUG_RETURN_CANVASA
    NSLog(@"🔍 DEBUG: Returning canvasA");
    return canvasA;
#endif
    
    // 3. Mask
    id<MTLTexture> maskA = [self createTexture:canvasWidth
                                        height:canvasHeight
                                        format:MTLPixelFormatR8Unorm];
    id<MTLTexture> maskB = [self createTexture:canvasWidth
                                        height:canvasHeight
                                        format:MTLPixelFormatR8Unorm];
    encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_maskPipeline];
    [encoder setTexture:canvasA atIndex:0];
    [encoder setTexture:warpB atIndex:1];
    [encoder setTexture:maskA atIndex:2];
    [encoder setTexture:maskB atIndex:3];
    [self dispatch:_maskPipeline encoder:encoder width:canvasWidth height:canvasHeight];
    [encoder endEncoding];
    [self debugTexture:warpB
                  name:@"WarpB"];
    
#ifdef DEBUG_RETURN_MASK
    NSLog(@"🔍 DEBUG: Returning maskA");
    return maskA;  // 或者返回 maskB 测试
#endif
    
    // 4. Distance
    id<MTLTexture> distanceA = [self createTexture:canvasWidth
                                           height:canvasHeight
                                           format:MTLPixelFormatR32Float];

    id<MTLTexture> distanceB = [self createTexture:canvasWidth
                                            height:canvasHeight
                                            format:MTLPixelFormatR32Float];
    [self runDistance:maskA output:distanceA commandBuffer:commandBuffer];
    [self runDistance:maskB output:distanceB commandBuffer:commandBuffer];
    
    // 5. Weight
    id<MTLTexture> weight = [self createTexture:canvasWidth
                                         height:canvasHeight
                                         format:MTLPixelFormatR32Float];
    encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_weightPipeline];
    [encoder setTexture:distanceA atIndex:0];
    [encoder setTexture:distanceB atIndex:1];
    [encoder setTexture:weight atIndex:2];
    [self dispatch:_weightPipeline encoder:encoder width:canvasWidth height:canvasHeight];
    [encoder endEncoding];
    
#ifdef DEBUG_RETURN_WEIGHT
    NSLog(@"🔍 DEBUG: Returning weight");
    return weight;
#endif
    
    // 6. 单层混合
    id<MTLTexture> result = [self simpleBlendA:canvasA B:warpB weight:weight commandBuffer:commandBuffer];
    [self debugTexture:textureA
                  name:@"BeforeBlend A"];


    [self debugTexture:warpB
                  name:@"BeforeBlend B"];


    [self debugTexture:result
                  name:@"Result Before"];
    return result;
}

#pragma mark - Distance Transform
- (void)runDistance:(id<MTLTexture>)mask output:(id<MTLTexture>)output commandBuffer:(id<MTLCommandBuffer>)commandBuffer
{
    NSUInteger w = mask.width, h = mask.height;
    id<MTLTexture> temp1 = [self createTexture:w
                                        height:h
                                        format:MTLPixelFormatR32Float];
    id<MTLTexture> temp2 = [self createTexture:w
                                        height:h
                                        format:MTLPixelFormatR32Float];
    
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_distanceInitPipeline];
    [encoder setTexture:mask atIndex:0];
    [encoder setTexture:temp1 atIndex:1];
    [self dispatch:_distanceInitPipeline encoder:encoder width:w height:h];
    [encoder endEncoding];
    
    uint32_t maxStep = 1;
    while (maxStep < MAX(w, h)) maxStep <<= 1;
    maxStep >>= 1;
    id<MTLTexture> src = temp1, dst = temp2;
    for (uint32_t step = maxStep; step >= 1; step >>= 1) {
        encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:_distanceJFAPipeline];
        [encoder setTexture:src atIndex:0];
        [encoder setTexture:dst atIndex:1];
        DistanceParams params;
        params.width = (uint32_t)w;
        params.height = (uint32_t)h;
        params.step = step;
        [encoder setBytes:&params length:sizeof(params) atIndex:0];
        [self dispatch:_distanceJFAPipeline encoder:encoder width:w height:h];
        [encoder endEncoding];
        id<MTLTexture> tmp = src; src = dst; dst = tmp;
    }
    if (src != output) {
        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        [blit copyFromTexture:src sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(w,h,1) toTexture:output destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
        [blit endEncoding];
    }
}

#pragma mark - Simple Blend (single level)
- (id<MTLTexture>)simpleBlendA:(id<MTLTexture>)A B:(id<MTLTexture>)B weight:(id<MTLTexture>)W commandBuffer:(id<MTLCommandBuffer>)cb
{
    id<MTLTexture> result = [self createTexture:A.width
                                         height:A.height
                                         format:MTLPixelFormatRGBA8Unorm];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:_blendPipeline];
    [enc setTexture:A atIndex:0];
    [enc setTexture:B atIndex:1];
    [enc setTexture:W atIndex:2];
    [enc setTexture:result atIndex:3];
    [self dispatch:_blendPipeline encoder:enc width:result.width height:result.height];
    [enc endEncoding];
    return result;
}

- (void)downSample:(id<MTLTexture>)input output:(id<MTLTexture>)output commandBuffer:(id<MTLCommandBuffer>)cb {}
- (void)upSample:(id<MTLTexture>)input output:(id<MTLTexture>)output commandBuffer:(id<MTLCommandBuffer>)cb {}
- (void)laplacian:(id<MTLTexture>)current up:(id<MTLTexture>)up output:(id<MTLTexture>)output commandBuffer:(id<MTLCommandBuffer>)cb {}
- (void)blendLaplacian:(id<MTLTexture>)A B:(id<MTLTexture>)B W:(id<MTLTexture>)W output:(id<MTLTexture>)output commandBuffer:(id<MTLCommandBuffer>)cb {}
- (void)reconstruct:(id<MTLTexture>)up lap:(id<MTLTexture>)lap output:(id<MTLTexture>)output commandBuffer:(id<MTLCommandBuffer>)cb {}

- (NSUInteger)outputWidth  { return _outputWidth; }
- (NSUInteger)outputHeight { return _outputHeight; }

@end
