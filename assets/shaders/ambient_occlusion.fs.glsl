#version 460

#extension GL_ARB_shading_language_include : require

#include "packing.glsl"

layout(location = 0) in vec2 v_uv;

layout(location = 0) out float f_ao;

layout(set = 0, binding = 0) uniform sampler float_sampler;
layout(set = 0, binding = 1) uniform sampler uint_sampler;
layout(set = 0, binding = 2) uniform utexture2D g_buffer;
layout(set = 0, binding = 3) uniform texture2D depth_buffer;

layout(set = 0, binding = 4) uniform Uniforms {
    mat4 inverse_projection;
    mat4 view;
    mat4 projection;
    float sample_count;
    float sample_radius;
    float intensity;
};

vec3 getPositionFromDepth(vec2 uv, float depth)
{
    vec4 pos = vec4(uv * 2.0 - 1.0, depth, 1.0);
    pos = inverse_projection * pos;
    return pos.xyz / pos.w;
}

vec3 getPosition(vec2 uv)
{
    // vec2 uv = v_pos.xy * 0.5 + vec2(0.5);
    float depth = texture(sampler2D(depth_buffer, float_sampler), uv).x;
    return getPositionFromDepth(uv, depth);
}

float InterleavedGradientNoise(vec2 position_screen)
{
    vec3 magic = vec3(0.0671106, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(position_screen, magic.xy)));
}

vec2 VogelDiskSample(int sampleIndex, float sqrtSamplesCount, float phi)
{
    const float GOLDEN_ANGLE = 2.4;
    float r = sqrt(sampleIndex + 0.5) / sqrtSamplesCount;
    float theta = sampleIndex*GOLDEN_ANGLE + phi;

    float sine = sin(theta);
    float cosine = cos(theta);
    return vec2(r*cosine, r*sine);
}

vec3 sampleOffset(int i) {
    // Generate pseudo-random values
    float r1 = InterleavedGradientNoise(vec2(i, 0.37));
    float r2 = InterleavedGradientNoise(vec2(i, 1.13));
    float r3 = InterleavedGradientNoise(vec2(i, 2.71));

    // Map to [-1,1]
    vec3 dir = normalize(vec3(r1 * 2.0 - 1.0, r2 * 2.0 - 1.0, r3));
    
    // Bias samples closer to center
    float scale = float(i) / sample_count;
    scale = mix(0.1, 1.0, scale * scale);

    return dir * scale;
}

const float bias = 0.025;

void main()
{
    float depth = texture(sampler2D(depth_buffer, float_sampler), v_uv).x;
    if (depth == 1.0) {
        f_ao = 1.0;
        return;
    }
    vec3 fragPos = getPositionFromDepth(v_uv, depth);
    uvec2 packed_color_and_normal = texture(usampler2D(g_buffer, uint_sampler), v_uv).rg;
    // vec3 position = getPosition(v_uv);
    vec3 normal = mat3(view) * DecodeNormal(unpackHalf2x16(packed_color_and_normal.g));
    vec3 randomVec = normalize(vec3(
        InterleavedGradientNoise(gl_FragCoord.xy) * 2.0 - 1.0,
        InterleavedGradientNoise(gl_FragCoord.xy + 81.0) * 2.0 - 1.0,
        0.0
    ));
    // create TBN change-of-basis matrix: from tangent-space to view-space
    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);
    // iterate over the sample kernel and calculate occlusion factor
    float occlusion = 0.0;
    for(int i = 0; i < sample_count; ++i)
    {
        // get sample position
        vec3 samplePos = TBN * sampleOffset(i); // from tangent to view-space
        samplePos = fragPos + samplePos * sample_radius; 
        
        // project sample position (to sample texture) (to get position on screen/texture)
        vec4 offset = vec4(samplePos, 1.0);
        offset = projection * offset; // from view to clip-space
        offset.xyz /= offset.w; // perspective divide
        offset.xyz = offset.xyz * 0.5 + 0.5; // transform to range 0.0 - 1.0
        
        // get sample depth
        float sampleDepth = getPosition(offset.xy).z;
        
        // range check & accumulate
        float rangeCheck = smoothstep(0.0, 1.0, sample_radius / abs(fragPos.z - sampleDepth));
        occlusion += (sampleDepth >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck;           
    }
    occlusion = 1.0 - (occlusion / sample_count);


    occlusion = mix(1.0, occlusion, intensity);
    
    f_ao = occlusion; //vec4(position, 1.0) * 0.0 + vec4(normal, 1.0);
}
