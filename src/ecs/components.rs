use std::sync::Arc;

use flecs_ecs::macros::Component;
use glam::{Quat, Vec3, Vec4};
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
    pub color_factor: Vec4,
    pub diffuse: Option<Arc<ImageView>>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub metallic_roughness: Option<Arc<ImageView>>,
    pub ambient_oclussion: Option<Arc<ImageView>>,
    pub emissive_factor: Vec3,
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
            color_factor: Vec4::ONE,
            diffuse: None,
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            metallic_roughness: None,
            ambient_oclussion: None,
            emissive_factor: Vec3::ZERO,
            emissive: None,
            normal: None,
        }
    }
}
