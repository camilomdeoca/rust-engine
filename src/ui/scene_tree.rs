use egui::Ui;
use flecs_ecs::{
    core::{Query, World},
    prelude::{Builder, QueryAPI, QueryBuilderImpl},
};
use glam::Vec3;

use crate::ecs::components::{
    DirectionalLight, DirectionalLightShadowMap, MaterialComponent, MeshComponent, PointLight,
    SceneEntity, Transform,
};

pub struct SceneTree {
    query: Query<()>,
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
        let query = world.query().with::<&SceneEntity>().build();
        Self { query }
    }

    pub fn draw(&mut self, ui: &mut Ui) {
        egui::ScrollArea::both().show(ui, |ui| {
            // Why: https://github.com/Indra-db/Flecs-Rust/issues/209
            self.query.run(|mut it| {
                while it.next() {
                    for i in it.iter() {
                        let entity = it.entity(i);
                        ui.collapsing(entity.name(), |ui| {
                            entity.try_get::<&mut Transform>(|transform| {
                                ui.collapsing("Transform", |ui| {
                                    edit_vec3_ui(ui, &mut transform.translation, "Position");
                                    ui.label(format!("Rotation: {}", transform.rotation));
                                    edit_vec3_ui(ui, &mut transform.scale, "Scale");
                                });
                            });

                            entity.try_get::<&MeshComponent>(|_mesh_component| {
                                ui.label("MeshComponent");
                            });

                            entity.try_get::<&MaterialComponent>(|_material_component| {
                                ui.label("MaterialComponent");
                            });

                            entity.try_get::<&mut PointLight>(|point_light| {
                                ui.collapsing("PointLight", |ui| {
                                    edit_vec3_ui(ui, &mut point_light.color, "Color");
                                    ui.add(
                                        egui::DragValue::new(&mut point_light.radius)
                                            .speed(0.1)
                                            .prefix("Radius: "),
                                    );
                                });
                            });

                            entity.try_get::<&mut DirectionalLight>(|directional_light| {
                                ui.collapsing("DirectionalLight", |ui| {
                                    edit_vec3_ui(ui, &mut directional_light.color, "Color");
                                });
                            });

                            if entity.has::<DirectionalLightShadowMap>() {
                                ui.label("DirectionalLightShadowMap");
                            }
                        });
                    }
                }
            });
        });
    }
}
