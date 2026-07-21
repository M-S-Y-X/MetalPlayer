//
//  Shaders.metal
//


#include <metal_stdlib>
using namespace metal;

// ---------- 顶点 / 片段着色器 ----------

typedef struct {
    float2 position;
    float2 texCoord;
} Vertex;

typedef struct {
    float4 position [[position]];
    float2 texCoord;
} VertexOut;

vertex VertexOut vertex_main(uint vertexID [[vertex_id]],
                             constant Vertex *vertices [[buffer(0)]]) {
    VertexOut out;
    out.position = float4(vertices[vertexID].position, 0.0, 1.0);
    out.texCoord = vertices[vertexID].texCoord;
    return out;
}

fragment float4 fragment_yuv_to_rgb(
    VertexOut in [[stage_in]],
    texture2d<float> yTexture [[texture(0)]],
    texture2d<float> uTexture [[texture(1)]],
    texture2d<float> vTexture [[texture(2)]])
{
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float y = yTexture.sample(s, in.texCoord).r;
    float u = uTexture.sample(s, in.texCoord).r - 0.5;
    float v = vTexture.sample(s, in.texCoord).r - 0.5;

    float r = y + 1.402 * v;
    float g = y - 0.344 * u - 0.714 * v;
    float b = y + 1.772 * u;

    return float4(r, g, b, 1.0);
}

// ---------- Lanczos 插值内核 ----------

constant int kLanczosRadius = 3;

float sinc(float x) {
    if (abs(x) < 1e-5) return 1.0;
    float pix = M_PI_F * x;
    return sin(pix) / pix;
}

float lanczos(float x, int a) {
    if (abs(x) >= float(a)) return 0.0;
    return sinc(x) * sinc(x / float(a));
}

kernel void lanczos_horizontal(
    texture2d<float, access::read>  inTexture  [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant uint4 &dims [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint srcW = dims.x;
    uint srcH = dims.y;
    uint dstW = dims.z;
    if (gid.x >= dstW || gid.y >= srcH) return;
    
    float srcX = (float(gid.x) + 0.5) * float(srcW) / float(dstW) - 0.5;
    int left   = int(floor(srcX)) - kLanczosRadius + 1;
    int right  = int(floor(srcX)) + kLanczosRadius;
    
    float result = 0.0;
    float weightSum = 0.0;
    for (int i = left; i <= right; ++i) {
        float weight = lanczos(srcX - float(i), kLanczosRadius);
        int clampedX = clamp(i, 0, int(srcW) - 1);
        float pixel = inTexture.read(uint2(clampedX, gid.y)).r;
        result += pixel * weight;
        weightSum += weight;
    }
    if (weightSum > 0.0) result /= weightSum;
    outTexture.write(float4(result, 0.0, 0.0, 1.0), gid);
}

kernel void lanczos_vertical(
    texture2d<float, access::read>  inTexture  [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant uint4 &dims [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint srcW = dims.x;
    uint srcH = dims.y;
    uint dstW = dims.z;
    uint dstH = dims.w;
    if (gid.x >= dstW || gid.y >= dstH) return;
    
    float srcY = (float(gid.y) + 0.5) * float(srcH) / float(dstH) - 0.5;
    int top    = int(floor(srcY)) - kLanczosRadius + 1;
    int bottom = int(floor(srcY)) + kLanczosRadius;
    
    float result = 0.0;
    float weightSum = 0.0;
    for (int j = top; j <= bottom; ++j) {
        float weight = lanczos(srcY - float(j), kLanczosRadius);
        int clampedY = clamp(j, 0, int(srcH) - 1);
        float pixel = inTexture.read(uint2(gid.x, clampedY)).r;
        result += pixel * weight;
        weightSum += weight;
    }
    if (weightSum > 0.0) result /= weightSum;
    outTexture.write(float4(result, 0.0, 0.0, 1.0), gid);
}

// ---------- 5x5 高斯下采样 ----------

kernel void downsample_blur(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant uint2 &srcSize [[buffer(0)]],
    constant uint2 &dstSize [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dstSize.x || gid.y >= dstSize.y) return;
    
    float scaleX = float(srcSize.x) / float(dstSize.x);
    float scaleY = float(srcSize.y) / float(dstSize.y);
    float2 srcCoord = (float2(gid) + 0.5) * float2(scaleX, scaleY);
    
    const float w[5] = {0.0545, 0.2442, 0.4026, 0.2442, 0.0545};
    float sum = 0.0;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            float2 samplePos = srcCoord + float2(dx, dy);
            samplePos = clamp(samplePos, 0.0, float2(srcSize) - 1.0);
            float pix = src.read(uint2(samplePos)).r;
            sum += pix * w[dx+2] * w[dy+2];
        }
    }
    dst.write(sum, gid);
}

// ---------- 误差计算与反投影 ----------

kernel void error_compute(
    texture2d<float, access::read>  originalLow [[texture(0)]],
    texture2d<float, access::read>  simulatedLow [[texture(1)]],
    texture2d<float, access::write> error [[texture(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= originalLow.get_width() || gid.y >= originalLow.get_height()) return;
    float orig = originalLow.read(gid).r;
    float sim  = simulatedLow.read(gid).r;
    error.write(orig - sim, gid);
}

kernel void upscale_error(
    texture2d<float, access::sample> errorLow [[texture(0)]],
    texture2d<float, access::write>  errorHigh [[texture(1)]],
    constant uint2 &highSize [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= highSize.x || gid.y >= highSize.y) return;
    float2 uv = (float2(gid) + 0.5) / float2(highSize);
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float e = errorLow.sample(s, uv).r;
    errorHigh.write(e, gid);
}

kernel void back_project(
    texture2d<float, access::read_write> highRes [[texture(0)]],
    texture2d<float, access::read>       errorHigh [[texture(1)]],
    constant float &lambda [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= highRes.get_width() || gid.y >= highRes.get_height()) return;
    float cur = highRes.read(gid).r;
    float e   = errorHigh.read(gid).r;
    highRes.write(cur + lambda * e, gid);
}

// ---------- 增强时域混合（含梯度一致性）----------

kernel void blend_temporal(
    texture2d<float, access::read>  curLanczos  [[texture(0)]],
    texture2d<float, access::read>  prevHigh    [[texture(1)]],
    texture2d<float, access::write> outInit     [[texture(2)]],
    constant float &fallbackWeight [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= curLanczos.get_width() || gid.y >= curLanczos.get_height()) return;
    
    float cur = curLanczos.read(gid).r;
    float prev = prevHigh.read(gid).r;
    
    float2 gradCur = 0, gradPrev = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            uint2 coord = uint2(clamp(int(gid.x)+dx, 0, int(curLanczos.get_width())-1),
                                clamp(int(gid.y)+dy, 0, int(curLanczos.get_height())-1));
            float c = curLanczos.read(coord).r;
            float p = prevHigh.read(coord).r;
            gradCur += float2(dx, dy) * c;
            gradPrev += float2(dx, dy) * p;
        }
    }
    float gradDiff = length(gradCur - gradPrev);
    
    float lumDiff = abs(cur - prev);
    float motion = clamp(lumDiff * 3.0 + gradDiff * 5.0, 0.0, 1.0);
    float weight = fallbackWeight * (1.0 - motion);
    
    float blended = cur * (1.0 - weight) + prev * weight;
    outInit.write(blended, gid);
}

// ---------- USM 锐化 ----------

kernel void usm_sharpen(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant float &intensity [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= src.get_width() || gid.y >= src.get_height()) return;
    
    float4 sum = 0.0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            uint2 coord = uint2(clamp(int(gid.x)+dx, 0, int(src.get_width())-1),
                                clamp(int(gid.y)+dy, 0, int(src.get_height())-1));
            sum += src.read(coord);
        }
    }
    float4 blurred = sum / 9.0;
    float4 sharp = src.read(gid);
    float4 result = sharp + (sharp - blurred) * intensity;
    dst.write(result, gid);
}

// ---------- 通用卷积 (支持任意奇数核大小) ----------
// 注意：变量名避免使用 half（保留关键字）

kernel void conv2d_relu(
    texture2d_array<float, access::sample>  inTexture  [[texture(0)]],
    texture2d_array<float, access::write>   outTexture [[texture(1)]],
    constant float*  weights [[buffer(0)]],
    constant float*  bias    [[buffer(1)]],
    constant uint4&  dims    [[buffer(2)]], // [inCh, outCh, width, height]
    constant uint&   kSize   [[buffer(3)]], // 核大小 (奇数)
    uint3 gid [[thread_position_in_grid]])
{
    uint outCh = dims.y, width = dims.z, height = dims.w;
    if (gid.x >= width || gid.y >= height || gid.z >= outCh) return;

    constexpr sampler s(address::clamp_to_zero, filter::nearest);
    uint inCh = dims.x;
    int kHalf = int(kSize) / 2;   // 修正：half → kHalf
    float sum = bias[gid.z];

    for (uint ic = 0; ic < inCh; ++ic) {
        uint wBase = (gid.z * inCh + ic) * (kSize * kSize);
        for (int dy = -kHalf; dy <= kHalf; ++dy) {
            for (int dx = -kHalf; dx <= kHalf; ++dx) {
                float2 pos = float2(gid.x + dx, gid.y + dy) + 0.5;
                float pix = inTexture.sample(s, pos, ic).r;
                uint wIdx = wBase + (dy + kHalf) * kSize + (dx + kHalf);
                sum += pix * weights[wIdx];
            }
        }
    }
    sum = max(sum, 0.0f);  // ReLU
    outTexture.write(float4(sum, 0, 0, 0), gid.xy, gid.z);
}

// ---------- 纹理相加 (2D) ----------

kernel void add_2d(
    texture2d<float, access::read>  texA [[texture(0)]],
    texture2d<float, access::read>  texB [[texture(1)]],
    texture2d<float, access::write> out  [[texture(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    float a = texA.read(gid).r;
    float b = texB.read(gid).r;
    out.write(float4(a + b, 0.0, 0.0, 1.0), gid);
}

// ---------- 双线性上采样 ----------

kernel void bilinear_upsample(
    texture2d<float, access::sample>  src [[texture(0)]],
    texture2d<float, access::write>   dst [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint outW = dst.get_width(), outH = dst.get_height();
    if (gid.x >= outW || gid.y >= outH) return;
    float2 uv = (float2(gid) + 0.5) / float2(outW, outH);
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float val = src.sample(s, uv).r;
    dst.write(float4(val, 0.0, 0.0, 1.0), gid);
}

// ---------- 安全转换：R32Float → R8Unorm ----------

kernel void convert_to_r8unorm(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float val = src.read(gid).r;
    val = clamp(val, 0.0f, 1.0f);
    dst.write(float4(val, 0.0, 0.0, 1.0), gid);
}

// ---------- 运动估计 (8x8 块匹配) ----------
kernel void block_motion_estimation(
    texture2d<float, access::read>  curTexture  [[texture(0)]],
    texture2d<float, access::read>  refTexture  [[texture(1)]],
    texture2d<uint, access::write>  motion      [[texture(2)]], // R: mv.x, G: mv.y
    constant uint& blockSize  [[buffer(0)]],
    constant uint& searchRange [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint2 blockCoord = gid * blockSize;
    float bestCost = 1e10;
    int2 bestMV = 0;

    for (int dy = -int(searchRange); dy <= int(searchRange); ++dy) {
        for (int dx = -int(searchRange); dx <= int(searchRange); ++dx) {
            float cost = 0.0;
            for (uint y = 0; y < blockSize; ++y) {
                for (uint x = 0; x < blockSize; ++x) {
                    uint2 curCoord = blockCoord + uint2(x, y);
                    uint2 refCoord = uint2(int(curCoord.x) + dx, int(curCoord.y) + dy);
                    curCoord = clamp(curCoord, uint2(0), uint2(curTexture.get_width()-1, curTexture.get_height()-1));
                    refCoord = clamp(refCoord, uint2(0), uint2(refTexture.get_width()-1, refTexture.get_height()-1));
                    float diff = curTexture.read(curCoord).r - refTexture.read(refCoord).r;
                    cost += diff * diff;
                }
            }
            if (cost < bestCost) {
                bestCost = cost;
                bestMV = int2(dx, dy);
            }
        }
    }
    uint encodedX = uint(bestMV.x + 32768);
    uint encodedY = uint(bestMV.y + 32768);
    motion.write(uint4(encodedX, encodedY, 0, 0), gid);
}

// ---------- 运动补偿 ----------
kernel void motion_compensate(
    texture2d<float, access::read>  refTexture [[texture(0)]],
    texture2d<uint, access::read>   motion     [[texture(1)]],
    texture2d<float, access::write> dstTexture [[texture(2)]],
    constant uint& blockSize [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint2 blockCoord = (gid / blockSize) * blockSize;
    uint2 mvRaw = motion.read(blockCoord / blockSize).rg;
    int2 mv = int2(int(mvRaw.x) - 32768, int(mvRaw.y) - 32768);
    int2 refCoord = int2(gid) + mv;
    refCoord = clamp(refCoord, int2(0), int2(refTexture.get_width()-1, refTexture.get_height()-1));
    float val = refTexture.read(uint2(refCoord)).r;
    dstTexture.write(float4(val, 0, 0, 0), gid);
}

// ---------- 非局部时域滤波 ----------
kernel void nonlocal_temporal_filter(
    texture2d<float, access::read>  curTexture  [[texture(0)]],
    texture2d<float, access::read>  aligned0    [[texture(1)]],
    texture2d<float, access::read>  aligned1    [[texture(2)]],
    texture2d<float, access::write> dstTexture  [[texture(3)]],
    constant float& decay [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    const int radius = 1;
    float cur = curTexture.read(gid).r;
    float sumWeight = 1.0;
    float sumPixel = cur;

    float weightHist = decay;
    for (int f = 0; f < 2; ++f) {
        texture2d<float, access::read> hist = (f == 0) ? aligned0 : aligned1;
        float bestDiff = 1e10;
        float bestVal = hist.read(gid).r;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                int2 sampleCoord = int2(gid) + int2(dx, dy);
                sampleCoord = clamp(sampleCoord, int2(0), int2(hist.get_width()-1, hist.get_height()-1));
                float histVal = hist.read(uint2(sampleCoord)).r;
                float diff = fabs(histVal - cur);
                if (diff < bestDiff) {
                    bestDiff = diff;
                    bestVal = histVal;
                }
            }
        }
        float similarity = exp(-bestDiff * 10.0);
        sumPixel += bestVal * similarity * weightHist;
        sumWeight += similarity * weightHist;
        weightHist *= decay;
    }
    float result = sumPixel / sumWeight;
    dstTexture.write(float4(result, 0, 0, 0), gid);
}

fragment float4 fragment_rgba_display(
    VertexOut in [[stage_in]],
    texture2d<float, access::sample> rgbaTexture [[texture(0)]])
{
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float4 color = rgbaTexture.sample(s, in.texCoord);
    
    // 如果颜色全黑，显示红色网格以便确认片段着色器在工作
    float isBlack = (color.r + color.g + color.b) < 0.001;
    if (isBlack) {
        float2 grid = fract(in.texCoord * 30.0);
        float gridLine = step(0.98, max(grid.x, grid.y));
        return float4(1.0, 0.0, 0.0, 1.0) * gridLine;
    }
    return color;
}
