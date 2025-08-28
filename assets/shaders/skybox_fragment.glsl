#version 450

#extension GL_ARB_shading_language_include : require

#include "packing.glsl"

layout(location = 0) in vec3 v_pos;

//layout(location = 0) out vec4 f_color;
layout(location = 0) out uvec2 g_buffer;

layout(set = 0, binding = 0) uniform sampler s;
layout(set = 0, binding = 1) uniform textureCube cubemap;

void main()
{
    vec3 color = texture(samplerCube(cubemap, s), v_pos).rgb;
    color = color / (color + vec3(1.0)); // tone mapping
    color = pow(color, vec3(1.0/2.2)); // gamme correction
    
    uint packed_color = pack11_10_11(color);
    g_buffer = uvec2(packed_color, 0.0);
}
