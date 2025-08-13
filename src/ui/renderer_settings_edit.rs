use std::sync::{Arc, RwLock};

use crate::renderer::Renderer;
use egui::Ui;

use super::edit_widgets::edit_array_ui;

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
            let label = ui.label("Shadow map bias");
            ui.add(
                egui::DragValue::new(&mut settings.shadow_map_base_bias)
                    .range(0.0..=1.0)
                    .speed(0.00001),
            )
            .labelled_by(label.id);
        });

        edit_array_ui(ui, &mut settings.sample_count_per_level);
    }
}
