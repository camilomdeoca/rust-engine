#version 460
//#extension GL_EXT_multiview : enable

layout(location = 0) in vec3 a_position;

struct EntityData {
    mat4 transform;
    uint material;
    uint pad[3];
};

struct DirectionalLight {
    vec3 direction;
    vec3 color;
    mat4 light_space;
};

layout(std430, set = 0, binding = 0) readonly buffer DirectionalLights {
    DirectionalLight directional_lights[];
};
layout(std430, set = 0, binding = 1) readonly buffer EntityDataBuffer {
    EntityData entity_data[];
};

void main() {
    mat4 model = entity_data[gl_DrawID].transform;
    vec4 position = model * vec4(a_position, 1.0);
    gl_Position = directional_lights[0].light_space * position;
    // vec4 position = model * vec4(a_position, 1.0);
    // gl_Position = lights_data[0/*gl_ViewIndex*/].light_space * position;
}
