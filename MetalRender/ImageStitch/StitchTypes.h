//
//  StitchTypes.h
//  MetalRender
//
//  Full Metal Stitch Types
//

#ifndef StitchTypes_h
#define StitchTypes_h

#import <Metal/Metal.h>
#import <simd/simd.h>

//======================================================
// 最大金字塔层数（对应 Python levels <=5）
//======================================================
#define STITCH_MAX_LEVEL 5

//======================================================
// Warp 参数（B -> Canvas）
// 注意：着色器期望的是逆矩阵 H_inv，CPU 端需预先求逆
//======================================================
typedef struct
{
    simd_float3x3 H_inv;

    uint width;
    uint height;

    int offsetX;
    int offsetY;
}WarpParams;

//======================================================
// Mask 参数（保留备用）
//======================================================
typedef struct
{
    uint32_t width;
    uint32_t height;
} MaskParams;

//======================================================
// Distance Transform 参数（Jump Flood Algorithm）
//======================================================
typedef struct
{
    uint32_t width;
    uint32_t height;
    uint32_t step;
} DistanceParams;

//======================================================
// Weight 参数（保留备用）
//======================================================
typedef struct
{
    float epsilon;
} WeightParams;

//======================================================
// Pyramid 参数（保留备用）
//======================================================
typedef struct
{
    uint32_t srcWidth;
    uint32_t srcHeight;
    uint32_t dstWidth;
    uint32_t dstHeight;
} PyramidParams;

//======================================================
// Blend 参数（保留备用）
//======================================================
typedef struct
{
    uint32_t width;
    uint32_t height;
} BlendParams;

#endif
