#version 460

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_tangent;
layout(location = 3) in vec2 a_uv;

layout(location = 0) out vec2 v_uv;

void main() {
    v_uv = a_position.xy * 0.5 + vec2(0.5);
    gl_Position = vec4(a_position, 1.0);
}
