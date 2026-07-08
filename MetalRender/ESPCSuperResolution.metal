#include <metal_stdlib>
using namespace metal;

// ==========================================
// 辅助函数
// ==========================================
inline float relu(float x) {
    return max(x, 0.0f);
}

inline float read_pixel_clamped(
    texture2d<float, access::read> tex,
    int2 pos,
    int width,
    int height
) {
    int2 clamped = int2(
        clamp(pos.x, 0, width - 1),
        clamp(pos.y, 0, height - 1)
    );
    return tex.read(uint2(clamped)).r;
}

// ==========================================
// 阶段1: 特征提取 (1ch→64ch, 5×5 conv)
// ==========================================
kernel void espcn_stage1_feature_extraction(
    texture2d<float, access::read>       yInput        [[texture(0)]],
    texture2d_array<float, access::write> feat1Out     [[texture(1)]],
    const device float                   *conv1_weight [[buffer(0)]],
    const device float                   *conv1_bias   [[buffer(1)]],
    uint2                                gid           [[thread_position_in_grid]]
) {
    int width = yInput.get_width();
    int height = yInput.get_height();
    
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;
    
    for (int oc = 0; oc < 64; oc++) {
        float sum = conv1_bias[oc];
        int w_offset = oc * 25;
        
        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                int2 sample_pos = int2(int(gid.x) + kx, int(gid.y) + ky);
                float pixel = read_pixel_clamped(yInput, sample_pos, width, height);
                sum += pixel * conv1_weight[w_offset + (ky + 2) * 5 + (kx + 2)];
            }
        }
        
        feat1Out.write(float4(relu(sum)), uint2(gid), uint(oc));
    }
}

// ==========================================
// 阶段2: 重建 (64→32→4, 3×3 conv)
// 修正：feat2Cache 从 read 改为 write 后不能再 read
// 解决方案：将所有计算放在前面，最后统一写入
// ==========================================
kernel void espcn_stage2_reconstruction(
    texture2d_array<float, access::read>  feat1Cache    [[texture(0)]],
    texture2d_array<float, access::write> feat2Cache    [[texture(1)]],
    texture2d_array<float, access::write> feat3Cache    [[texture(2)]],
    const device float                    *conv2_weight [[buffer(0)]],
    const device float                    *conv2_bias   [[buffer(1)]],
    const device float                    *conv3_weight [[buffer(2)]],
    const device float                    *conv3_bias   [[buffer(3)]],
    uint2                                 gid           [[thread_position_in_grid]]
) {
    int width = feat1Cache.get_width();
    int height = feat1Cache.get_height();
    
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;
    
    // ======== 第二层: 64→32, 3×3, ReLU ========
    // 先全部计算到本地数组
    float feat2[32];
    for (int oc = 0; oc < 32; oc++) {
        float sum = conv2_bias[oc];
        
        for (int ic = 0; ic < 64; ic++) {
            int w_offset = (oc * 64 + ic) * 9;
            
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int sx = clamp(int(gid.x) + kx, 0, width - 1);
                    int sy = clamp(int(gid.y) + ky, 0, height - 1);
                    
                    // feat1Cache 是 read 权限，可以读取
                    float4 val4 = feat1Cache.read(uint2(sx, sy), uint(ic));
                    sum += val4.x * conv2_weight[w_offset + (ky + 1) * 3 + (kx + 1)];
                }
            }
        }
        
        feat2[oc] = relu(sum);
    }
    
    // 一次性写入 feat2Cache
    for (int oc = 0; oc < 32; oc++) {
        feat2Cache.write(float4(feat2[oc]), uint2(gid), uint(oc));
    }
    
    // ======== 第三层: 32→4, 3×3 ========
    for (int oc = 0; oc < 4; oc++) {
        float sum = conv3_bias[oc];
        
        for (int ic = 0; ic < 32; ic++) {
            int w_offset = (oc * 32 + ic) * 9;
            
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int sx = clamp(int(gid.x) + kx, 0, width - 1);
                    int sy = clamp(int(gid.y) + ky, 0, height - 1);
                    
                    // 使用本地 feat2 数组，而不是从纹理读取
                    sum += feat2[ic] * conv3_weight[w_offset + (ky + 1) * 3 + (kx + 1)];
                }
            }
        }
        
        feat3Cache.write(float4(sum), uint2(gid), uint(oc));
    }
}

// ==========================================
// 阶段3: Pixel Shuffle (Depth to Space ×2)
// ==========================================
kernel void espcn_stage3_pixel_shuffle(
    texture2d_array<float, access::read> feat3Cache [[texture(0)]],
    texture2d<float, access::write>      yOutput    [[texture(1)]],
    uint2                                gid        [[thread_position_in_grid]]
) {
    int out_width = yOutput.get_width();
    int out_height = yOutput.get_height();
    
    if (gid.x >= uint(out_width) || gid.y >= uint(out_height)) return;
    
    int2 input_pos = int2(gid) / 2;
    int channel = (int(gid.y) % 2) * 2 + (int(gid.x) % 2);
    
    float4 val4 = feat3Cache.read(uint2(input_pos), uint(channel));
    yOutput.write(val4.x, uint2(gid));
}
