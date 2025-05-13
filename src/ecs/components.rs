use flecs_ecs::macros::Component;
use glam::{Quat, Vec3};

use crate::assets::database::MeshId;

#[derive(Debug, Component)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

#[derive(Debug, Component)]
pub struct Mesh {
    pub id: MeshId,
}

