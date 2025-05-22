#version 450

layout(location = 0) in vec3 v_pos;

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler s;
layout(set = 0, binding = 1) uniform textureCube cubemap;

void main()
{
    vec3 color = texture(samplerCube(cubemap, s), v_pos).rgb;
    color = color / (color + vec3(1.0)); // tone mapping
    color = pow(color, vec3(1.0/2.2)); // gamme correction
    f_color = vec4(color, 1.0);
}
