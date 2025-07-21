#version 460

layout(constant_id = 0) const uint MAX_LIGHTS_PER_TILE = 256;
layout(constant_id = 1) const uint TILE_SIZE = 16;
layout(constant_id = 2) const uint Z_SLICES = 32;

layout (local_size_x = 64) in;

struct PointLight {
    vec3 position;
    float radius;
    vec3 color;
    uint pad[1];
};

layout(set = 0, binding = 0) uniform FrameUniforms {
    mat4 view;
    mat4 view_proj;
    uint num_lights;
    float width;
    float height;
    float near;
    float far;

};
layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight point_lights[];
};
// The first is a counter, it should be a global variable shared between thread groups
layout(std430, set = 0, binding = 2) buffer NextLigthIndexGlobal {
    uint next_global_light_index; // Next global index
};
layout(std430, set = 0, binding = 3) writeonly buffer VisibleLightIndices {
    uint light_indices[];
};
layout(set = 0, binding = 4, rg32ui) writeonly uniform uimage3D light_grid;

float start_depth_for_nth_slice(uint slice)
{
    return near * pow(far / near, float(slice) / float(Z_SLICES));
}
shared uint index_offset_out;
shared uint light_list[MAX_LIGHTS_PER_TILE];
shared uint next_index;

void main()
{
    if (gl_LocalInvocationID.x == 0)
    {
        next_index = 0;
    }

    barrier();

    vec4 planes[6];

    ivec2 tileID = ivec2(gl_WorkGroupID.xy);
    ivec2 tileNumber = ivec2(gl_NumWorkGroups.xy);
    
    {
        float minDepth = start_depth_for_nth_slice(gl_WorkGroupID.z);
        float maxDepth = start_depth_for_nth_slice(gl_WorkGroupID.z + 1);

		// Steps based on tile sale
		vec2 negativeStep = (2.0 * vec2(tileID)) / vec2(tileNumber);
		vec2 positiveStep = (2.0 * vec2(tileID + ivec2(1, 1))) / vec2(tileNumber);

		// Set up starting values for planes using steps and min and max z values
		planes[0] = vec4( 1.0,  0.0,  0.0,  1.0 - negativeStep.x); // Left
		planes[1] = vec4(-1.0,  0.0,  0.0, -1.0 + positiveStep.x); // Right
		planes[2] = vec4( 0.0,  1.0,  0.0,  1.0 - negativeStep.y); // Bottom
		planes[3] = vec4( 0.0, -1.0,  0.0, -1.0 + positiveStep.y); // Top
		planes[4] = vec4( 0.0,  0.0, -1.0, -minDepth); // Near
		planes[5] = vec4( 0.0,  0.0,  1.0,  maxDepth); // Far

		// Transform the first four planes
		for (uint i = 0; i < 4; i++) {
			planes[i] *= view_proj;
			planes[i] /= length(planes[i].xyz);
		}

		// Transform the depth planes
		planes[4] *= view;
		planes[4] /= length(planes[4].xyz);
		planes[5] *= view;
		planes[5] /= length(planes[5].xyz);
    }

	// Time to loop through the lights and cull them based on distance & plane frustums,
	// Compute shader runs once for each texel, need to calculate the index, we do this using 
	// the index of the thread relative to the working group
	for (uint i = gl_LocalInvocationID.x; i < num_lights; i += gl_WorkGroupSize.x)
	{
		PointLight light = point_lights[i];

		// We need to transform the light's position from world to view space ( *NOT* view_projection ),
		// radius remains the same since no scaling is occurring, assuming position is already is world coordinates
		// Rotation doesn't make sense for point lights and scaling is done via its radius
		vec4 position = vec4(light.position, 1.0); // Not going into projective space

		// // Get light's luminance using Rec 709 luminance formula
		// float light_luminance = dot(light.color, vec3(0.2126, 0.7152, 0.0722));
		// // Minimum luminance threshold - tweak to taste
		// float min_luminance = 0.5;
		// float radius = sqrt(light_luminance / min_luminance - 1.0); // light.radius;

        bool is_inside = true;

		// We check if the light exists in our frustum
		for (uint j = 0; j < 6; j++) {
			float distance = dot(position, planes[j]) + light.radius;

			// If one of the tests fails, then there is no intersection
			if (distance <= 0.0) {
				is_inside = false;
			}
		}

		// If light is inside, time to add it
		if (is_inside)
		{
			// Reserving index and writing light
            uint index = atomicAdd(next_index, 1);
            if (index >= MAX_LIGHTS_PER_TILE)
            {
                break;
            }
            else
            {
			    light_list[index] = i; // Just writing index
            }
		}
	}

    barrier();


	// Now time for each group ( first thread of each group ) to write its results 
	// to the global list
    if (gl_LocalInvocationID.x == 0)
    {
        next_index = min(next_index, MAX_LIGHTS_PER_TILE); // If one thread went past max it breaks
                                                           // without storing the light

	    index_offset_out = atomicAdd(next_global_light_index, next_index);
	    
	    imageStore(
            light_grid,
            ivec3(gl_WorkGroupID),
            uvec4(index_offset_out, next_index, 0u, 0u)
        );
    }

    barrier();

	// Let's take advantage of the single thread per pixel again, this time we just copy the light indices in the final light indices
	for (uint i = gl_LocalInvocationID.x; i < next_index; i += gl_WorkGroupSize.x)
    {
		light_indices[index_offset_out + i] = light_list[i];
    }
}
