use glam::{Quat, Vec3};

#[derive(Clone)]
pub struct Camera {
    pub position: Vec3,
    pub rotation: Quat,
    pub fov: f32,
}
