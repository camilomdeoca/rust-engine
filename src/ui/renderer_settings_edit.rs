use std::sync::{Arc, RwLock};

use crate::renderer::Renderer;
use egui::Ui;

use super::edit_widgets::ArrayDragEdit;

pub struct RendererSettingsEdit {
    renderer: Arc<RwLock<Renderer>>,
}

impl RendererSettingsEdit {
    pub fn new(renderer: Arc<RwLock<Renderer>>) -> Self {
        Self { renderer }
    }

    pub fn draw(&mut self, ui: &mut Ui) {
        let mut renderer_write = self.renderer.write().unwrap();
        let settings = renderer_write.settings_mut();

        ui.horizontal(|ui| {
            let label = ui.label("Light culling tile size: ");
            ui.add(
                egui::DragValue::new(&mut settings.light_culling_tile_size)
                    .range(1..=u32::max_value()),
            )
            .labelled_by(label.id);
        });

        ui.horizontal(|ui| {
            let label = ui.label("Light culling z slices: ");
            ui.add(
                egui::DragValue::new(&mut settings.light_culling_z_slices)
                    .range(1..=u32::max_value()),
            )
            .labelled_by(label.id);
        });

        ui.horizontal(|ui| {
            let label = ui.label("Light culling max lights per tile");
            ui.add(
                egui::DragValue::new(&mut settings.light_culling_max_lights_per_tile)
                    .range(1..=u32::max_value()),
            )
            .labelled_by(label.id);
        });

        ui.horizontal(|ui| {
            let label = ui.label("Cascaded shadow map level size");
            ui.add(
                egui::DragValue::new(&mut settings.cascaded_shadow_map_level_size)
                    .range(128..=u32::max_value())
                    .speed(0.25),
            )
            .labelled_by(label.id);
            settings.cascaded_shadow_map_level_size =
                settings.cascaded_shadow_map_level_size & !(128 - 1);
        });
        
        ui.horizontal(|ui| {
            let label = ui.label("Shadow bias");
            ui.add(
                egui::DragValue::new(&mut settings.shadow_bias)
                    .range(0.0..=1.0)
                    .speed(0.00001),
            )
            .labelled_by(label.id);
            // ui.separator();
            let label = ui.label("slope bias");
            ui.add(
                egui::DragValue::new(&mut settings.shadow_slope_bias)
                    .range(0.0..=1.0)
                    .speed(0.00001),
            )
            .labelled_by(label.id);
            let label = ui.label("normal bias");
            ui.add(
                egui::DragValue::new(&mut settings.shadow_normal_bias)
                    .range(0.0..=1.0)
                    .speed(0.00001),
            )
            .labelled_by(label.id);
        });
        
        ui.horizontal(|ui| {
            let label = ui.label("Penumbra max size");
            ui.add(
                egui::DragValue::new(&mut settings.penumbra_max_size)
                    .range(0.0..=1.0)
                    .speed(0.00001),
            )
            .labelled_by(label.id);
        });
        ui.horizontal(|ui| {
            let label = ui.label("Shadow map cascade split lambda");
            ui.add(
                egui::DragValue::new(&mut settings.shadow_map_cascade_split_lambda)
                    .range(0.0..=1.0)
                    .speed(0.001),
            )
            .labelled_by(label.id);
        });

        ui.collapsing("Ambient occlusion", |ui| {
            ui.horizontal(|ui| {
                let label = ui.label("Sample count");
                ui.add(
                    egui::DragValue::new(&mut settings.sample_count)
                        .range(0..=128)
                        .speed(0.1),
                )
                .labelled_by(label.id);
            });
            ui.horizontal(|ui| {
                let label = ui.label("Sample radius");
                ui.add(
                    egui::DragValue::new(&mut settings.sample_radius)
                        .range(0.0..=30.0)
                        .speed(0.01),
                )
                .labelled_by(label.id);
            });
            ui.horizontal(|ui| {
                let label = ui.label("Intensity");
                ui.add(
                    egui::DragValue::new(&mut settings.ambient_occlusion_intensity)
                        .range(0.0..=10.0)
                        .speed(0.01),
                )
                .labelled_by(label.id);
            });
        });

        ui.add(ArrayDragEdit::new(&mut settings.sample_count_per_level));
    }
}
