#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

struct VertexIn {
    float2 position;
    float2 texCoord;
};

vertex VertexOut vertex_main(
    uint vertexID [[vertex_id]],
    constant VertexIn *vertices [[buffer(0)]]
) {
    VertexOut out;
    out.position = float4(vertices[vertexID].position, 0.0, 1.0);
    out.texCoord = vertices[vertexID].texCoord;
    return out;
}

fragment float4 fragment_yuv_to_rgb(
    VertexOut in [[stage_in]],
    texture2d<float, access::sample> yTexture [[texture(0)]],
    texture2d<float, access::sample> uTexture [[texture(1)]],
    texture2d<float, access::sample> vTexture [[texture(2)]]
) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    
    float y = yTexture.sample(s, in.texCoord).r;
    float u = uTexture.sample(s, in.texCoord).r - 0.5;
    float v = vTexture.sample(s, in.texCoord).r - 0.5;
    
    float r = y + 1.402 * v;
    float g = y - 0.344 * u - 0.714 * v;
    float b = y + 1.772 * u;
    
    return float4(r, g, b, 1.0);
}
