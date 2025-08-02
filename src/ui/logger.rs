use std::sync::{LazyLock, RwLock};

use egui::{Color32, TextStyle, Ui};
use log::{Level, LevelFilter, Metadata, Record};

struct LoggerData {
    lines: Vec<(Level, String)>,
}

impl Default for LoggerData {
    fn default() -> Self {
        Self {
            lines: vec![]
        }
    }
}

static LOG_LINES: LazyLock<RwLock<LoggerData>> = LazyLock::new(|| {
    RwLock::new(LoggerData::default())
});

pub struct LoggerImplementation {}

impl log::Log for LoggerImplementation {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= LevelFilter::Debug
    }

    fn log(&self, record: &Record) {
        LOG_LINES.write().unwrap().lines.push((record.level(), record.args().to_string()));
    }

    fn flush(&self) {
        todo!()
    }
}

const fn level_color(level: &Level) -> Color32 {
    match level {
        Level::Error => Color32::LIGHT_RED,
        Level::Warn => Color32::YELLOW,
        Level::Info => Color32::CYAN,
        Level::Debug => Color32::MAGENTA,
        Level::Trace => Color32::WHITE,
    }
}

pub fn ui(ui: &mut Ui) {
    egui::ScrollArea::both()
        .stick_to_bottom(true)
        .show(ui, |ui| {
            let mut job = egui::text::LayoutJob::default();

            let style = ui.ctx().style();
            let font_id = TextStyle::Monospace.resolve(&style);

            for (level, text) in &LOG_LINES.read().unwrap().lines {
                let color = level_color(level);
                let text_format_for_level_of_line = egui::TextFormat {
                    font_id: font_id.clone(),
                    color,
                    background: Color32::TRANSPARENT,
                    ..Default::default()
                };
                job.append("[", 0.0, text_format_for_level_of_line.clone());
                job.append(level.as_str(), 0.0, text_format_for_level_of_line.clone());
                job.append("] ", 0.0, text_format_for_level_of_line.clone());
                job.append(&text, 0.0, egui::TextFormat {
                    font_id: font_id.clone(),
                    color: style.visuals.text_color(),
                    background: Color32::TRANSPARENT,
                    ..Default::default()
                });
                job.append("\n", 0.0, egui::TextFormat {
                    font_id: font_id.clone(),
                    ..Default::default()
                });

            }

            let galley = ui.fonts(|f| f.layout_job(job));
            let (response, painter) =
                ui.allocate_painter(galley.rect.size(), egui::Sense::click_and_drag());

            if response.clicked() && !response.has_focus() {
                ui.memory_mut(|mem| mem.request_focus(response.id));
            }

            // painter.rect_filled(
            //     gallery.rect.translate(response.rect.min.to_vec2()),
            //     0.0,
            //
            // );

            painter.galley(response.rect.min, galley, Color32::DEBUG_COLOR);
            ui.ctx().set_cursor_icon(egui::CursorIcon::Text);
        });
}
