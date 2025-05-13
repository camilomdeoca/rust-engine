#version 450

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec2 v_uv;

layout(set = 0, binding = 0) uniform CameraUniforms {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

layout(set = 1, binding = 0) uniform ModelUniforms {
    mat4 transform;
} model;

void main() {
    mat4 worldview = uniforms.view * uniforms.world;
    v_normal = transpose(inverse(mat3(worldview))) * a_normal;
    v_uv = a_uv;
    gl_Position = uniforms.proj * worldview * model.transform * vec4(a_position, 1.0);
}
