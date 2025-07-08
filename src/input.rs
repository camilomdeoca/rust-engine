use std::collections::HashMap;

use flecs_ecs::core::Entity;
use flecs_ecs::prelude::*;
use glam::{Quat, Vec2};
use winit::{event::{DeviceEvent, ElementState, RawKeyEvent}, keyboard::{KeyCode, PhysicalKey}};

use crate::ecs::components::Transform;

struct InputManager {
    keys_state: HashMap<KeyCode, bool>,
    accumulated_delta_movement: Vec2,
}

impl InputManager {
    fn new() -> Self {
        Self {
            keys_state: HashMap::new(),
            accumulated_delta_movement: Vec2::ZERO,
        }
    }

    fn process_event(&mut self, event: DeviceEvent) {
        match event {
            DeviceEvent::Key(RawKeyEvent {
                physical_key: PhysicalKey::Code(key_code),
                state,
            }) => {
                self.keys_state.insert(key_code, state == ElementState::Pressed);
            }
            DeviceEvent::MouseMotion { delta } => {
                self.accumulated_delta_movement.x += delta.0 as f32;
                self.accumulated_delta_movement.y += delta.1 as f32;
            }
            DeviceEvent::MouseWheel { delta: _delta } => {
                // TODO: zoom
            }
            _ => {}
        }
    }

    // fn (&mut self, EntityView) {
    //     self.controlled_entity = entity;
    // }
}
