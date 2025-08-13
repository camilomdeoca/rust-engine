#version 460

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_tangent;
layout(location = 3) in vec2 a_uv;

layout(location = 0) out vec3 v_pos;
layout(location = 1) out float v_view_space_depth;
layout(location = 2) out vec2 v_uv;
layout(location = 3) flat out int v_draw_id;
layout(location = 4) out mat3 v_TBN;

const uint SHADOW_MAP_CASCADE_COUNT = 4; // Cant be specialization constant if we still want
                                         // to have vulkano-shaders do all the nice things

struct EntityData {
    mat4 transform;
    uint material;
    uint pad[3];
};

// Frame descriptor set
//   - changes every frame
layout(set = 1, binding = 0) uniform FrameUniforms {
    mat4 view;
    mat4 proj;
    vec3 view_position;
    float near;
    float far;
    float width;
    float height;
    uint directional_light_count;
    float cutoff_distances[SHADOW_MAP_CASCADE_COUNT];
    uint light_culling_tile_size;
    uint light_culling_z_slices;
    uint sample_count_per_level[SHADOW_MAP_CASCADE_COUNT];
    float shadow_map_base_bias;
    mat4 light_space_matrices[SHADOW_MAP_CASCADE_COUNT];
};
layout(std430, set = 1, binding = 1) readonly buffer EntityDataBuffer {
    EntityData entity_data[];
};

void main() {
    mat4 model = entity_data[gl_DrawID].transform;
    v_draw_id = gl_DrawID;

    vec3 T = normalize(vec3(model * vec4(a_tangent, 0.0)));
    vec3 N = normalize(vec3(model * vec4(a_normal, 0.0)));
    vec3 B = normalize(cross(N, T));
    v_TBN = mat3(T, B, N);
    v_uv = a_uv;

    vec4 position = model * vec4(a_position, 1.0);
    vec4 view_space_position = view * position;
    v_view_space_depth = view_space_position.z/view_space_position.w;
    gl_Position = proj * view_space_position;
    v_pos = position.xyz/position.w;
}
