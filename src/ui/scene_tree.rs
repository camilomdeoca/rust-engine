use egui::Ui;
use flecs_ecs::{core::{Query, World}, prelude::{Builder, QueryAPI, QueryBuilderImpl}};

use crate::ecs::components::{MaterialComponent, MeshComponent, SceneEntity};

pub struct SceneTree {
    query: Query<(Option<&'static MeshComponent>, Option<&'static MaterialComponent>)>,
}

impl SceneTree {
    pub fn new(world: World) -> Self {
        let query = world
            .query::<(Option<&MeshComponent>, Option<&MaterialComponent>)>()
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
                self.query.each_entity(|entity, (mesh_component, material_component)| {
                    ui.collapsing(entity.name(),|ui| {
                        if let Some(_mesh_component) = mesh_component {
                            ui.label("MeshComponent");
                        }
                        
                        if let Some(_material_component) = material_component {
                            ui.label("MaterialComponent");
                        }
                    });
                });
            });
    }
}
