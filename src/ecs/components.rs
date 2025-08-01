use flecs_ecs::{core::EntityView, macros::Component};
use glam::{Quat, Vec2, Vec3};

use crate::assets::database::{CubemapId, MaterialId, MeshId};

#[derive(Debug, Component)]
pub struct SceneEntity;

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

#[derive(Debug, Component)]
pub struct PointLight {
    pub color: Vec3,
    pub radius: f32,
}

#[derive(Debug, Component)]
pub struct DirectionalLight {
    pub color: Vec3,
}

#[derive(Debug, Component)]
pub struct DirectionalLightShadowMap;

#[derive(Debug, Component)]
pub struct Camera {
    pub fov: f32,
}

#[derive(Debug, Component)]
pub struct Controllable {
    pub on_mouse_move: fn(EntityView, Vec2),
}
