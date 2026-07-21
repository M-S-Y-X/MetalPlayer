//
//  Shaders_Stitch.metal
//

#include <metal_stdlib>
using namespace metal;

//======================================================
// Struct
//======================================================
struct WarpParams
{
    float3x3 H_inv;
    uint width;
    uint height;
    int offsetX;
    int offsetY;
};

struct DistanceParams
{
    uint width;
    uint height;
    uint step;
};

//======================================================
// Bilinear Sample
//======================================================
float4 sampleBilinear(
    texture2d<float, access::read> tex,
    float2 p)
{
    uint w = tex.get_width();
    uint h = tex.get_height();

    // 使用 clamp 避免边缘异常
    float x = clamp(p.x, 0.0f, float(w - 1));
    float y = clamp(p.y, 0.0f, float(h - 1));

    int x0 = floor(x);
    int y0 = floor(y);
    int x1 = min(x0 + 1, int(w - 1));
    int y1 = min(y0 + 1, int(h - 1));
    float fx = x - float(x0);
    float fy = y - float(y0);

    float4 c00 = tex.read(uint2(x0, y0));
    float4 c10 = tex.read(uint2(x1, y0));
    float4 c01 = tex.read(uint2(x0, y1));
    float4 c11 = tex.read(uint2(x1, y1));

    return c00 * (1 - fx) * (1 - fy) + c10 * fx * (1 - fy) + c01 * (1 - fx) * fy + c11 * fx * fy;
}

//======================================================
// 1. Perspective Warp（使用逆矩阵）
//======================================================
kernel void warpPerspectiveKernel(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant WarpParams &param [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    
    if(gid.x >= param.width ||
       gid.y >= param.height)
    {
        return;
    }
    
    
    // canvas坐标
    float3 dst = float3(gid.x, gid.y, 1.0);
    
    
    // 转回真实拼接坐标
    dst.x -= param.offsetX;
    dst.y -= param.offsetY;
    
    
    // H_inv : A坐标 -> B坐标
    float3 src = param.H_inv * dst;
    
    
    src.x /= src.z;
    src.y /= src.z;
    
    
    // 判断是否在B图范围
    
    if(src.x < 0 ||
       src.y < 0 ||
       src.x >= input.get_width()-1 ||
       src.y >= input.get_height()-1)
    {
        output.write(float4(0,0,0,0),gid);
        return;
    }
    
    
    float4 color = sampleBilinear(input,src.xy);
    
    output.write(color,gid);
}


//======================================================
// 2. Mask
//======================================================
kernel void generateMaskKernel(texture2d<float, access::read> A[[texture(0)]],
                               texture2d<float, access::read> B[[texture(1)]],
                               texture2d<float, access::write> maskA[[texture(2)]],
                               texture2d<float, access::write> maskB[[texture(3)]],
                               uint2 gid [[thread_position_in_grid]])
{
    if(gid.x>=A.get_width() || gid.y>=A.get_height())
        return;

    float4 ca = A.read(gid);
    float4 cb = B.read(gid);
    float ma = any(ca.rgb > 0.001) ? 1.0 : 0.0;
    float mb = any(cb.rgb > 0.001) ? 1.0 : 0.0;

    maskA.write(float4(ma), gid);
    maskB.write(float4(mb), gid);
}

//======================================================
// 3. Distance Init
//======================================================
kernel void distanceInitKernel
(texture2d<float, access::read> mask[[texture(0)]],
 texture2d<float, access::write> distance[[texture(1)]],
 uint2 gid[[thread_position_in_grid]])
{
    if(gid.x>=mask.get_width() || gid.y>=mask.get_height())
        return;

    float m = mask.read(gid).r;
    float value = m > 0.5 ? 0 : 99999;
    distance.write(float4(value), gid);
}

//======================================================
// 4. Distance JFA Step
//======================================================
kernel void distanceJFAKernel(texture2d<float, access::read> input[[texture(0)]],
                              texture2d<float, access::write> output[[texture(1)]],
                              constant DistanceParams &param[[buffer(0)]],
                              uint2 gid[[thread_position_in_grid]])
{
    if(gid.x>=param.width || gid.y>=param.height)
        return;

    float best = input.read(gid).r;
    int step = param.step;

    for(int y = -1; y <= 1; y++)
    {
        for(int x = -1; x <= 1; x++)
        {
            int2 p = int2(gid) + int2(x,y) * step;
            if(p.x >= 0 && p.y >= 0 && p.x < param.width && p.y < param.height)
            {
                float d = input.read(uint2(p)).r;
                best = min(best,d+step);
            }
        }
    }

    output.write(float4(best), gid);
}

//======================================================
// 5. Weight
//======================================================
kernel void weightKernel(texture2d<float, access::read> A[[texture(0)]], texture2d<float, access::read> B[[texture(1)]], texture2d<float, access::write> output[[texture(2)]], uint2 gid[[thread_position_in_grid]])
{
    if(gid.x>=output.get_width() || gid.y>=output.get_height())
        return;

    float da = A.read(gid).r;
    float db = B.read(gid).r;

    float w = db / (da + db + 0.00001);

    output.write(float4(w), gid);
}

//======================================================
// 6. Gaussian Down
//======================================================
kernel void gaussianDownKernel(texture2d<float, access::read> input[[texture(0)]],
                               texture2d<float, access::write> output[[texture(1)]],
                               uint2 gid[[thread_position_in_grid]])
{
    if(gid.x >= output.get_width() || gid.y >= output.get_height())
        return;

    uint2 src = gid * 2;
    float4 sum = 0;

    for(int y = -2; y <= 2; y++)
    {
        for(int x = -2; x <= 2; x++)
        {
            int sx = clamp(int(src.x) + x, 0, int(input.get_width()) - 1);
            int sy = clamp(int(src.y) + y, 0, int(input.get_height()) - 1);
            sum += input.read(uint2(sx,sy)) * 0.04;
        }
    }

    output.write(sum, gid);
}

//======================================================
// 7. Gaussian Up
//======================================================
kernel void gaussianUpKernel(texture2d<float, access::read> input[[texture(0)]],
                             texture2d<float, access::write> output[[texture(1)]],
                             uint2 gid[[thread_position_in_grid]])
{
    float2 p = float2(gid) / 2.0;
    float4 c = sampleBilinear(input, p);
    output.write(c * 4, gid);
}

//======================================================
// 8. Laplacian
//======================================================
kernel void laplacianKernel(texture2d<float, access::read> current[[texture(0)]],
                            texture2d<float, access::read> up[[texture(1)]],
                            texture2d<float, access::write> output[[texture(2)]],
                            uint2 gid[[thread_position_in_grid]])
{
    float4 a = current.read(gid);
    float4 b = up.read(uint2(min(gid.x, up.get_width() - 1), min(gid.y, up.get_height() - 1)));
    output.write(a - b, gid);
}

//======================================================
// 9. Pyramid Blend
//======================================================
kernel void pyramidBlendKernel(texture2d<float, access::read> A[[texture(0)]],
                               texture2d<float, access::read> B[[texture(1)]],
                               texture2d<float, access::read> W[[texture(2)]],
                               texture2d<float, access::write> output[[texture(3)]],
                               uint2 gid[[thread_position_in_grid]])
{
    float w = W.read(gid).r;
    float4 a = A.read(gid);
    float4 b = B.read(gid);
    output.write(a * w + b * (1 - w), gid);
}

//======================================================
// 10. Reconstruction
//======================================================
kernel void reconstructKernel(texture2d<float, access::read> up[[texture(0)]],
                              texture2d<float, access::read> lap[[texture(1)]],
                              texture2d<float, access::write> output[[texture(2)]],
                              uint2 gid[[thread_position_in_grid]])
{
    float4 result = up.read(gid) + lap.read(gid);
    output.write(clamp(result, 0.0, 1.0), gid);
}

