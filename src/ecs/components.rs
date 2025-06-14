use flecs_ecs::macros::Component;
use glam::{Quat, Vec3};

use crate::assets::database::{CubemapId, MaterialId, MeshId};

#[derive(Debug, Component)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

#[derive(Debug, Component)]
pub struct MeshComponent {
    pub mesh_id: MeshId,
}

#[derive(Debug, Component, Clone)]
pub struct MaterialComponent {
    pub material_id: MaterialId,
}

#[derive(Debug, Component)]
pub struct EnvironmentCubemap {
    pub environment_map: CubemapId,
    pub irradiance_map: CubemapId,
    pub prefiltered_environment_map: CubemapId,
    pub environment_brdf_lut: CubemapId,
}
