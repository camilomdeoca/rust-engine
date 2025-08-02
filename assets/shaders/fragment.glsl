#version 460

#extension GL_EXT_nonuniform_qualifier : require

layout(constant_id = 0) const uint TILE_SIZE = 16;
layout(constant_id = 1) const uint Z_SLICES = 32;

const uint SHADOW_MAP_CASCADE_COUNT = 3; // Cant be specialization constant if we still want
                                         // to have vulkano-shaders do all the nice things

layout(location = 0) in vec3 v_pos;
layout(location = 1) in float v_view_space_depth;
layout(location = 2) in vec2 v_uv;
layout(location = 3) flat in int v_draw_id;
layout(location = 4) in mat3 v_TBN;

layout(location = 0) out vec4 f_color;

struct Material {
    vec4 base_color_factor;
    uint base_color_texture_id;
    float metallic_factor;
    float roughness_factor;
    uint metallic_roughness_texture_id;
    uint ambient_oclussion_texture_id;
    vec3 emissive_factor;
    uint emissive_texture_id;
    uint normal_texture_id;
    uint pad[3];
};

struct EntityData {
    mat4 transform;
    uint material;
    uint pad[3];
};

struct PointLight {
    vec3 position;
    float radius;
    vec3 color;
    uint pad[1];
};

struct DirectionalLight {
    vec3 direction;
    vec3 color;
    bool has_shadow_maps;
};

// Slow changing descriptor set 
//   - doesn't change every frame
//   - changes when a texture is added (a material is added)
layout(set = 0, binding = 0) uniform sampler s;
layout(set = 0, binding = 1) uniform textureCube irradiance_map;
layout(set = 0, binding = 2) uniform textureCube prefiltered_environment_map;
layout(set = 0, binding = 3) uniform texture2D environment_brdf_lut;
layout(std430, set = 0, binding = 4) readonly buffer MaterialBuffer {
    Material materials[];
};
layout(set = 0, binding = 5) uniform texture2D textures[];

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
    mat4 light_space_matrices[SHADOW_MAP_CASCADE_COUNT];
};
layout(std430, set = 1, binding = 1) readonly buffer EntityDataBuffer {
    EntityData entity_data[];
};
layout(std430, set = 1, binding = 2) readonly buffer DirectionalLights {
    DirectionalLight directional_lights[];
};
layout(std430, set = 1, binding = 3) readonly buffer PointLights {
    PointLight point_lights[];
};
layout(std430, set = 1, binding = 4) readonly buffer VisibleLightIndices {
    uint light_indices[];
};
layout(set = 1, binding = 5, rg32ui) readonly uniform uimage3D light_grid;
layout(set = 1, binding = 6) uniform texture2DArray shadow_map;

const uint UINT_MAX = 4294967295;

const float PI = 3.14159265359;
// ----------------------------------------------------------------------------
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float k = (roughness*roughness) / 2.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    denom = max(denom, 0.000001);

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
// ----------------------------------------------------------------------------
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}   
// ----------------------------------------------------------------------------

uint slice_for_view_space_depth(float depth)
{
    float log_far_near = log2(far / near);
    float scale = float(Z_SLICES) / log_far_near;
    float bias = - (float(Z_SLICES) * log2(near)) / log_far_near;
    return uint(floor(log2(depth) * scale + bias));
}

vec3 get_light_contribution(
    vec3 base_color,
    vec3 radiance,
    vec3 light_direction,
    vec3 view_direction,
    vec3 normal,
    float metallic,
    float roughness,
    vec3 F0
) {
    vec3 H = normalize(view_direction + light_direction);

    // Cook-Torrance BRDF
    float NDF = DistributionGGX(normal, H, roughness);
    float G   = GeometrySmith(normal, view_direction, light_direction, roughness);      
    vec3  F   = fresnelSchlick(clamp(dot(H, view_direction), 0.0, 1.0), F0);
       
    vec3 numerator    = NDF * G * F;
    float denominator = 4.0
        * max(dot(normal, view_direction), 0.0)
        * max(dot(normal, light_direction), 0.0)
        + 0.0001; // + 0.0001 to prevent divide by zero
    vec3 specular = numerator / denominator;
    
    // kS is equal to Fresnel
    vec3 kS = F;
    // for energy conservation, the diffuse and specular light can't
    // be above 1.0 (unless the surface emits light); to preserve this
    // relationship the diffuse component (kD) should equal 1.0 - kS.
    vec3 kD = vec3(1.0) - kS;
    // multiply kD by the inverse metalness such that only non-metals
    // have diffuse lighting, or a linear blend if partly metal (pure metals
    // have no diffuse light).
    kD *= 1.0 - metallic;

    // scale light by NdotL
    float NdotL = max(dot(normal, light_direction), 0.0);

    // add to outgoing radiance Lo
    return (kD * base_color / PI + specular) * radiance * NdotL; // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
}

float texture_project(vec3 projCoords, uint level_index, float bias)
{
    // transform to [0,1] range
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(sampler2DArray(shadow_map, s), vec3(projCoords.xy, level_index)).r; 
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    // check whether current frag pos is in shadow
    float shadow;
    if (projCoords.x >= 0.0 && projCoords.x <= 1.0 && projCoords.y >= 0.0 && projCoords.y <= 1.0)
    {
        shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;
        // f_color = vec4(closestDepth, currentDepth, max(closestDepth, currentDepth), 1.0);
        // return;
    }
    else
        shadow = 0.0;

    return shadow;
}

void main()
{
    vec4 base_color;
    float metallic, roughness;
    float ao;
    vec3 emissive;
    vec3 N;

    uint material_id = entity_data[v_draw_id].material;
    Material material = materials[material_id];

    if (material.base_color_texture_id != UINT_MAX)
    {
        base_color =
            texture(
                nonuniformEXT(sampler2D(textures[material.base_color_texture_id], s)),
                v_uv
            )
            * material.base_color_factor;
    }
    else
    {
        base_color = material.base_color_factor;
    }

    if (base_color.a < 0.0001) discard; // TODO: draw transparent meshes after opaque

    if (material.metallic_roughness_texture_id != UINT_MAX)
    {
        vec2 aux =
            texture(
                nonuniformEXT(sampler2D(textures[material.metallic_roughness_texture_id], s)),
                v_uv
            ).gb;

        metallic = aux.g * material.metallic_factor;
        roughness = aux.r * material.roughness_factor;
    }
    else
    {
        metallic = material.metallic_factor;
        roughness = material.roughness_factor;
    }

    if (material.ambient_oclussion_texture_id != UINT_MAX)
    {
        ao =
            texture(
                nonuniformEXT(sampler2D(textures[material.ambient_oclussion_texture_id], s)),
                v_uv
            ).r;
    }
    else
    {
        ao = 1.0;
    }

    if (material.emissive_texture_id != UINT_MAX)
    {
        emissive =
            texture(
                nonuniformEXT(sampler2D(textures[material.emissive_texture_id], s)),
                v_uv
            ).rgb
            * material.emissive_factor.rgb;
    }
    else
    {
        emissive = material.emissive_factor.rgb;
    }

    if (material.normal_texture_id != UINT_MAX)
    {
        N =
            texture(
                nonuniformEXT(sampler2D(textures[material.normal_texture_id], s)),
                v_uv
            ).rgb;
    }
    else
    {
        N = vec3(0.0, 0.0, 1.0);
    }
    N = N * 2.0 - 1.0;
    N = normalize(v_TBN * N);
    vec3 V = normalize(view_position - v_pos);
    vec3 R = reflect(-V, N);
    
    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)    
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, base_color.rgb, metallic);

	uvec2 tile = uvec2(gl_FragCoord.xy / vec2(float(TILE_SIZE)));
    uint slice = slice_for_view_space_depth(-v_view_space_depth);

	uvec2 num_tiles = uvec2(ceil(width / float(TILE_SIZE)), ceil(height / float(TILE_SIZE)));
	uint index =
        + slice * num_tiles.y * num_tiles.x
        + tile.y * num_tiles.x
        + tile.x;

    // Calculating tile index from it
    uvec2 light_grid_data = imageLoad(light_grid, ivec3(tile, slice)).xy;
	uint num_lights = light_grid_data.y;
	uint light_offset = light_grid_data.x;

    // reflectance equation
    vec3 Lo = vec3(0.0);

    for (int i = 0; i < directional_light_count; i++)
    {
        DirectionalLight light = directional_lights[i];
        // calculate per-light radiance
        vec3 light_direction = normalize(-light.direction);
        vec3 radiance = light.color;

        float shadow = 0.0;
        if (light.has_shadow_maps)
        {
            uint level = 0;
            for (uint i = 0; i < SHADOW_MAP_CASCADE_COUNT - 1; i++)
            {
                if (v_view_space_depth < cutoff_distances[i])
                {
                    level = i + 1;
                }
            }

            const mat4 bias_mat = mat4( 
                0.5, 0.0, 0.0, 0.0,
                0.0, 0.5, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.5, 0.5, 0.0, 1.0
            );
            vec4 pos_light_space = bias_mat * light_space_matrices[level] * vec4(v_pos, 1.0);
            vec3 projected_coords = pos_light_space.xyz / pos_light_space.w;

            float bias = 0.00025;
            bias = max(bias * (1.0 - dot(N, light_direction)), bias);

            shadow = texture_project(projected_coords, level, bias);
            // f_color = vec4(1.0, shadow * 0.0, 0.0, 1.0);
            // return;
		    // switch(level) {
		    // 	case 0 : 
		    // 		f_color = vec4(1.0, 0.25, 0.25, 1.0 + shadow*0.0);
		    // 		break;
		    // 	case 1 : 
		    // 		f_color = vec4(0.25, 1.0, 0.25, 1.0);
		    // 		break;
		    // 	case 2 : 
		    // 		f_color = vec4(0.25, 0.25, 1.0, 1.0);
		    // 		break;
		    // }
		    //       return;
        }
        // add to outgoing radiance Lo
        Lo += (1.0 - shadow) * get_light_contribution(base_color.rgb, radiance, light_direction, V, N, metallic, roughness, F0);
    }

    for (int i = 0; i < num_lights; i++)
    {
        PointLight light = point_lights[light_indices[light_offset + i]];
        // calculate per-light radiance
        vec3 light_direction = normalize(light.position - v_pos);
        float distance = length(light.position - v_pos);
        float attenuation = 1.0 / (1.0 + distance * distance);
        vec3 radiance = light.color * attenuation;

        // add to outgoing radiance Lo
        Lo += get_light_contribution(base_color.rgb, radiance, light_direction, V, N, metallic, roughness, F0);
    }

    vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    
    // ambient lighting (we now use IBL as the ambient term)
    vec3 kS = fresnelSchlick(max(dot(N, V), 0.0), F0);
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;
    vec3 irradiance = texture(samplerCube(irradiance_map, s), N).rgb;
    vec3 diffuse    = irradiance * base_color.rgb;
    
    // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(samplerCube(prefiltered_environment_map, s), R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(sampler2D(environment_brdf_lut, s), vec2(max(dot(N, V), 0.0), roughness)).rg;
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

    vec3 ambient = (kD * diffuse + specular) * ao;

    vec3 color = ambient + Lo + emissive;

    // HDR tonemapping
    color = color / (color + vec3(1.0));

    // gamma correct TODO: Make it toggleable because egui expects no gamma correction but when
    // rendering to the screen it needs to be working
    //color = pow(color, vec3(1.0/2.2)); 

    f_color = vec4(color, 1.0);

    // float intensity = num_lights / (64 / 2.0);
    // f_color = vec4(vec3(intensity, intensity * 0.5, intensity * 0.5) + f_color.rgb * 0.25, 1.0); //light culling debug
    if (num_lights >= 64)
        f_color = vec4(1.0, 0.0, 0.0, 1.0);

    // vec3 colors[] = {
    //     vec3(0.0, 1.0, 0.0),
    //     vec3(0.0, 0.0, 1.0),
    //     vec3(1.0, 0.0, 0.0),
    //     vec3(1.0, 1.0, 1.0),
    //     vec3(0.0, 1.0, 1.0),
    //     vec3(1.0, 1.0, 0.0),
    //     vec3(1.0, 0.0, 1.0),
    //     vec3(0.0, 0.0, 0.0),
    // };
    // f_color.rgb = colors[slice % 8];

    // For debugging
    // if (uint(gl_FragCoord.x) % 16 == 0 || uint(gl_FragCoord.y) % 16 == 0) {
    //     f_color.rgb = mix(f_color.rgb, vec3(1.0), 0.25);
    // } else
    // if (num_lights > 0) {
    //     switch (num_lights) {
    //         case 1:  f_color.rgb = mix(f_color.rgb, vec3(0.0, 1.0, 0.0), 0.25); break;
    //         case 2:  f_color.rgb = mix(f_color.rgb, vec3(0.0, 1.0, 1.0), 0.25); break;
    //         case 3:  f_color.rgb = mix(f_color.rgb, vec3(0.0, 0.0, 1.0), 0.25); break;
    //         case 4:  f_color.rgb = mix(f_color.rgb, vec3(1.0, 0.0, 1.0), 0.25); break;
    //         default: f_color.rgb = mix(f_color.rgb, vec3(1.0, 0.0, 0.0), 0.25); break;
    //     }
    // }
}
