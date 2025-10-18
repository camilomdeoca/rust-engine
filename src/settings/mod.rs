
pub const SHADOW_MAP_CASCADE_COUNT: u32 = 4;

#[derive(Clone)]
pub struct ShadowMappingSettings {
    pub cascade_level_size: u32,
    pub sample_count_per_level: [u32; SHADOW_MAP_CASCADE_COUNT as usize],
    pub bias: f32,
    pub slope_bias: f32,
    pub normal_bias: f32,
    pub penumbra_max_size: f32,
    pub cascade_split_lambda: f32,
}

#[derive(Clone)]
pub struct LightCullingSettings {
    pub max_lights_per_tile: u32,
    pub tile_size: u32,
    pub z_slices: u32,
}

#[derive(Clone)]
pub struct AmbientOcclusionSettings {
    pub sample_count: u32,
    pub sample_radius: f32,
    pub ambient_occlusion_intensity: f32,
}

#[derive(Clone)]
pub struct RendererSettings {
    pub light_culling: LightCullingSettings,
    pub shadow_mapping: ShadowMappingSettings,
    pub ambient_occlusion: AmbientOcclusionSettings,
}

#[derive(Clone)]
pub struct Settings {
    pub renderer: RendererSettings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            renderer: RendererSettings {
                light_culling: LightCullingSettings {
                    max_lights_per_tile: 64,
                    tile_size: 32,
                    z_slices: 32,
                },
                shadow_mapping: ShadowMappingSettings {
                    cascade_level_size: 1536,
                    sample_count_per_level: [10, 8, 2, 2],
                    bias: 0.0005,
                    slope_bias: 0.0005,
                    normal_bias: 0.012,
                    penumbra_max_size: 0.0015,
                    cascade_split_lambda: 0.75,
                },
                ambient_occlusion: AmbientOcclusionSettings {
                    sample_count: 16,
                    sample_radius: 0.5,
                    ambient_occlusion_intensity: 1.0,
                },
            },
        }
    }
}
