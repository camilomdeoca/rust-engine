#version 450

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_tangent;
layout(location = 3) in vec2 a_uv;

layout(location = 0) out vec3 v_pos;

layout(set = 1, binding = 0) uniform FrameUniforms {
    mat4 view_proj; // view without translation
} uniforms;

void main() {
    v_pos = a_position;
    const vec4 position = uniforms.view_proj * vec4(a_position, 1.0);
    gl_Position = position.xyww;
}
