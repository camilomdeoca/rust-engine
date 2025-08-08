#version 460
#extension GL_EXT_multiview : enable

const uint SHADOW_MAP_LEVELS = 4;

layout(location = 0) in vec3 a_position;

struct EntityData {
    mat4 transform;
    uint material;
    uint pad[3];
};

layout(set = 0, binding = 0) uniform LightData {
    mat4 light_space_matrices[SHADOW_MAP_LEVELS];
};

layout(std430, set = 0, binding = 1) readonly buffer EntityDataBuffer {
    EntityData entity_data[];
};

void main() {
    mat4 model = entity_data[gl_DrawID].transform;
    vec4 position = model * vec4(a_position, 1.0);
    gl_Position = light_space_matrices[gl_ViewIndex] * position;
    // vec4 position = model * vec4(a_position, 1.0);
    // gl_Position = lights_data[0/*gl_ViewIndex*/].light_space * position;
}
