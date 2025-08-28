#version 460

#extension GL_ARB_shading_language_include : require

#include "packing.glsl"

layout(location = 0) in vec2 v_uv;

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler s;
layout(set = 0, binding = 1) uniform utexture2D g_buffer;
layout(set = 0, binding = 2) uniform texture2D ambient_occlusion_buffer;

void main()
{
    uvec2 packed_color_and_normal = texture(usampler2D(g_buffer, s), v_uv).rg;
    vec3 color = unpack11_10_11(packed_color_and_normal.r);
    vec3 normal = DecodeNormal(unpackHalf2x16(packed_color_and_normal.g));
    float ambient_occlusion = texture(sampler2D(ambient_occlusion_buffer, s), v_uv).r;
    f_color = vec4((normal * 0.5 + 0.5) * 0.0 + color * 0.0 + vec3(ambient_occlusion), 1.0);
}
