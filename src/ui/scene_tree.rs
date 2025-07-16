use egui::Ui;
use flecs_ecs::{core::{Query, World}, prelude::{Builder, QueryAPI, QueryBuilderImpl}};
use glam::Vec3;

use crate::ecs::components::{MaterialComponent, MeshComponent, PointLight, SceneEntity, Transform};

pub struct SceneTree {
    query: Query<(
        Option<&'static mut Transform>,
        Option<&'static mut MeshComponent>,
        Option<&'static mut MaterialComponent>,
        Option<&'static mut PointLight>,
    )>,
}

fn edit_vec3_ui(ui: &mut egui::Ui, vec: &mut Vec3, label: &str) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::DragValue::new(&mut vec.x).speed(0.1).prefix("x: "));
        ui.add(egui::DragValue::new(&mut vec.y).speed(0.1).prefix("y: "));
        ui.add(egui::DragValue::new(&mut vec.z).speed(0.1).prefix("z: "));
    });
}

impl SceneTree {
    pub fn new(world: World) -> Self {
        let query = world
            .query()
            .with::<&SceneEntity>()
            .build();
        Self {
            query,
        }
    }

    pub fn draw(
        &mut self,
        ui: &mut Ui,
    ) {
        egui::ScrollArea::both()
            .show(ui, |ui| {
                self.query.each_entity(|entity, (transform, mesh_component, material_component, point_light)| {
                    ui.collapsing(entity.name(),|ui| {
                        if let Some(transform) = transform {
                            ui.collapsing("Transform", |ui| {
                                edit_vec3_ui(ui, &mut transform.translation, "Position");
                                ui.label(format!("Rotation: {}", transform.rotation));
                                edit_vec3_ui(ui, &mut transform.scale, "Scale");
                            });
                        }
                        
                        if let Some(_mesh_component) = mesh_component {
                            ui.label("MeshComponent");
                        }
                        
                        if let Some(_material_component) = material_component {
                            ui.label("MaterialComponent");
                        }
                        
                        if let Some(point_light) = point_light {
                            ui.collapsing("PointLight", |ui| {
                                edit_vec3_ui(ui, &mut point_light.color, "Color");
                                ui.add(
                                    egui::DragValue::new(&mut point_light.radius)
                                        .speed(0.1)
                                        .prefix("Radius: ")
                                );
                            });
                        }
                    });
                });
            });
    }
}
