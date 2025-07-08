#version 460

#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec3 v_pos;
layout(location = 1) in vec2 v_uv;
layout(location = 2) flat in int v_draw_id;
layout(location = 3) in mat3 v_TBN;
layout(location = 6) in vec3 v_light_pos;

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
    mat3 inv_view;
    mat4 proj;
};
layout(std430, set = 1, binding = 1) readonly buffer EntityDataBuffer {
    EntityData entity_data[];
};

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

    vec3 light_color = vec3(3000.0);
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
    vec3 V = normalize(-v_pos);
    vec3 R = reflect(-V, N);
    
    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)    
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, base_color.rgb, metallic);

    // reflectance equation
    vec3 Lo = vec3(0.0);
    {
        // calculate per-light radiance
        vec3 L = normalize(v_light_pos - v_pos);
        vec3 H = normalize(V + L);
        float distance = length(v_light_pos - v_pos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = light_color * attenuation;

        // Cook-Torrance BRDF
        float NDF = DistributionGGX(N, H, roughness);   
        float G   = GeometrySmith(N, V, L, roughness);      
        vec3 F    = fresnelSchlick(clamp(dot(H, V), 0.0, 1.0), F0);
           
        vec3 numerator    = NDF * G * F; 
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero
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
        float NdotL = max(dot(N, L), 0.0);        

        // add to outgoing radiance Lo
        Lo += (kD * base_color.rgb / PI + specular) * radiance * NdotL;  // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
    }

    vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    
    // ambient lighting (we now use IBL as the ambient term)
    vec3 kS = fresnelSchlick(max(dot(N, V), 0.0), F0);
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;
    vec3 Nuvw = inv_view * N;
    Nuvw.x *= -1.0;
    vec3 irradiance = texture(samplerCube(irradiance_map, s), Nuvw).rgb;
    vec3 diffuse      = irradiance * base_color.rgb;
    
    // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 Ruvw = inv_view * R;
    Ruvw.x *= -1.0;
    vec3 prefilteredColor = textureLod(samplerCube(prefiltered_environment_map, s), Ruvw, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 brdf  = texture(sampler2D(environment_brdf_lut, s), vec2(max(dot(N, V), 0.0), roughness)).rg;
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

    vec3 ambient = (kD * diffuse + specular) * ao;

    vec3 color = ambient + Lo + emissive;

    // HDR tonemapping
    color = color / (color + vec3(1.0));

    // gamma correct TODO: Make it toggleable because egui expects no gamma correction but when
    // rendering to the screen it needs to be working
    //color = pow(color, vec3(1.0/2.2)); 

    f_color = vec4(color, base_color.a);
}
