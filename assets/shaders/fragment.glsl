#version 450

layout(location = 0) in vec3 v_pos;
layout(location = 1) in vec2 v_uv;
layout(location = 2) in mat3 v_TBN;
layout(location = 5) in vec3 v_light_pos;

layout(location = 0) out vec4 f_color;

// Environment descriptor
layout(set = 0, binding = 0) uniform sampler s;
layout(set = 0, binding = 1) uniform textureCube irradiance_map;
layout(set = 0, binding = 2) uniform textureCube prefiltered_environment_map;
layout(set = 0, binding = 3) uniform texture2D environment_brdf_lut;

// Frame descriptor set
layout(set = 1, binding = 0) uniform FrameUniforms {
    mat4 view;
    mat3 inv_view;
    mat4 proj;
} frame;

// Material descriptor
layout(set = 2, binding = 0) uniform texture2D diffuse;
layout(set = 2, binding = 1) uniform texture2D metallic_roughness;
layout(set = 2, binding = 2) uniform texture2D ambient_oclussion;
layout(set = 2, binding = 3) uniform texture2D emissive;
layout(set = 2, binding = 4) uniform texture2D normal;

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
    vec3 light_color = vec3(3000.0);
    vec3 N = texture(sampler2D(normal, s), v_uv).rgb;
    N = N * 2.0 - 1.0;
    N = normalize(v_TBN * N);
    vec3 V = normalize(-v_pos);
    vec3 R = reflect(-V, N);
    
    vec3 albedo = texture(sampler2D(diffuse, s), v_uv).rgb;
    float metallic = texture(sampler2D(metallic_roughness, s), v_uv).b;
    float roughness = texture(sampler2D(metallic_roughness, s), v_uv).g;
    float ao = texture(sampler2D(ambient_oclussion, s), v_uv).r;
    vec3 emissive = texture(sampler2D(emissive, s), v_uv).rgb;

    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)    
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);

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
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;  // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
    }

    vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    
    // ambient lighting (we now use IBL as the ambient term)
    vec3 kS = fresnelSchlick(max(dot(N, V), 0.0), F0);
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;
    vec3 Nuvw = frame.inv_view * N;
    Nuvw.x *= -1.0;
    vec3 irradiance = texture(samplerCube(irradiance_map, s), Nuvw).rgb;
    vec3 diffuse      = irradiance * albedo;
    
    // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 Ruvw = frame.inv_view * R;
    Ruvw.x *= -1.0;
    vec3 prefilteredColor = textureLod(samplerCube(prefiltered_environment_map, s), Ruvw, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 brdf  = texture(sampler2D(environment_brdf_lut, s), vec2(max(dot(N, V), 0.0), roughness)).rg;
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

    vec3 ambient = (kD * diffuse + specular) * ao;

    vec3 color = ambient + Lo + emissive;

    // HDR tonemapping
    color = color / (color + vec3(1.0));
    // gamma correct
    color = pow(color, vec3(1.0/2.2)); 

    f_color = vec4(color, 1.0);
}
