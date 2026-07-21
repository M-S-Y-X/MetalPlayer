//
//  ImageStitch.h
//
//  Full Metal Stitch Engine
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <simd/simd.h>

NS_ASSUME_NONNULL_BEGIN

@interface ImageStitcher : NSObject


/**
 初始化 Stitch Engine
 @param device Metal Device
 */
- (instancetype)initWithDevice:(id<MTLDevice>)device;


/**
 完整图像拼接
 输入: textureA:第一张图
     textureB:第二张图
     homography:B -> A 的单应矩阵
    注意：    外部需要提前计算好
    等价 Python:    H_new
 canvasWidth:    输出画布宽
 canvasHeight:    输出画布高
 offsetX:    xMin偏移
 offsetY:    yMin偏移
 返回:拼接后的 Texture
 流程: TextureA
       |
    TextureB
       |
    Warp
       |
    Mask
       |
 Distance Feather
       |
 Multi Band Blend
       |
    Output
 */
- (nullable id<MTLTexture>)stitchTextureA:(id<MTLTexture>)textureA textureB:(id<MTLTexture>)textureB homography:(simd_float3x3)H canvasWidth:(NSUInteger)canvasWidth canvasHeight:(NSUInteger)canvasHeight offsetX:(int)offsetX offsetY:(int)offsetY commandBuffer:(id<MTLCommandBuffer>)commandBuffer;

/**
 当前输出尺寸
 */
@property(nonatomic,readonly)NSUInteger outputWidth;
@property(nonatomic,readonly)NSUInteger outputHeight;

@end

NS_ASSUME_NONNULL_END
