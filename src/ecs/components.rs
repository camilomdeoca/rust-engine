use std::sync::Arc;

use flecs_ecs::macros::Component;
use glam::{Quat, Vec3};
use vulkano::{buffer::Subbuffer, image::view::ImageView};

use crate::assets::vertex::Vertex;

#[derive(Debug, Component)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

#[derive(Debug, Component)]
pub struct Mesh {
    pub vertex_buffer: Subbuffer<[Vertex]>,
    pub index_buffer: Subbuffer<[u32]>,
}

#[derive(Debug, Component, Clone)]
pub struct Material {
    pub diffuse: Option<Arc<ImageView>>,
    pub metallic_roughness: Option<Arc<ImageView>>,
    pub ambient_oclussion: Option<Arc<ImageView>>,
    pub emissive: Option<Arc<ImageView>>,
    pub normal: Option<Arc<ImageView>>,
}

#[derive(Debug, Component)]
pub struct EnvironmentCubemap {
    pub environment_map: Arc<ImageView>,
    pub irradiance_map: Arc<ImageView>,
    pub prefiltered_environment_map: Arc<ImageView>,
    pub environment_brdf_lut: Arc<ImageView>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            diffuse: None,
            metallic_roughness: None,
            ambient_oclussion: None,
            emissive: None,
            normal: None,
        }
    }
}
