#version 460

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_tangent;
layout(location = 3) in vec2 a_uv;

layout(location = 0) out vec3 v_pos;
layout(location = 1) out vec2 v_uv;
layout(location = 2) flat out int v_draw_id;
layout(location = 3) out mat3 v_TBN;
layout(location = 6) out vec3 v_light_pos;

struct EntityData {
    mat4 transform;
    uint material;
    uint pad[3];
};

// Frame descriptor set
//   - changes every frame
layout(set = 1, binding = 0) uniform FrameUniforms {
    mat4 view;
    mat3 inv_view;
    mat4 proj;
};
layout(std430, set = 1, binding = 1) readonly buffer EntityDataBuffer {
    EntityData entity_data[];
};

void main() {
    mat4 view_model = view * entity_data[gl_DrawID].transform;
    v_draw_id = gl_DrawID;

    vec3 T = normalize(vec3(view_model * vec4(a_tangent, 0.0)));
    vec3 N = normalize(vec3(view_model * vec4(a_normal, 0.0)));
    vec3 B = normalize(cross(N, T));
    v_TBN = mat3(T, B, N);
    v_uv = a_uv;

    vec4 position = view_model * vec4(a_position, 1.0);
    gl_Position = proj * position;
    v_pos = position.xyz/position.w;

    vec4 light_pos = view * vec4(-10.0, 10.0, 20.0, 1.0);
    v_light_pos = light_pos.xyz/light_pos.w;
}
