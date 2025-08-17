use egui::Ui;
use flecs_ecs::{
    core::{Query, World},
    prelude::{Builder, QueryAPI, QueryBuilderImpl},
};

use crate::ecs::components::{
    DirectionalLight, DirectionalLightShadowMap, MaterialComponent, MeshComponent, PointLight,
    SceneEntity, Transform,
};

use super::edit_widgets::{QuatDragEditAsEulerAngles, Vec3DragEdit};

pub struct SceneTree {
    query: Query<()>,
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
                                    let label = ui.label("Position");
                                    ui.add(Vec3DragEdit::new(&mut transform.translation))
                                        .labelled_by(label.id);
                                    let label = ui.label("Rotation");
                                    ui.add(QuatDragEditAsEulerAngles::new(&mut transform.rotation))
                                        .labelled_by(label.id);
                                    let label = ui.label("Scale");
                                    ui.add(Vec3DragEdit::new(&mut transform.scale))
                                        .labelled_by(label.id);
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
                                    let label = ui.label("Color");
                                    ui.add(Vec3DragEdit::new(&mut point_light.color))
                                        .labelled_by(label.id);
                                    let label = ui.label("Radius");
                                    ui.add(
                                        egui::DragValue::new(&mut point_light.radius).speed(0.1),
                                    )
                                    .labelled_by(label.id);
                                });
                            });

                            entity.try_get::<&mut DirectionalLight>(|directional_light| {
                                ui.collapsing("DirectionalLight", |ui| {
                                    let label = ui.label("Color");
                                    ui.add(Vec3DragEdit::new(&mut directional_light.color))
                                        .labelled_by(label.id);
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
