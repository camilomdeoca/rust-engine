use std::{sync::{Arc, RwLock}, time::Duration};

use camera_view::CameraView;
use egui::{Frame, Margin};
use egui_tiles::Tree;
use egui_winit_vulkano::Gui;
use flecs_ecs::core::World;
use scene_tree::SceneTree;
use renderer_settings_edit::RendererSettingsEdit;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    image::
        view::ImageView
    ,
    memory::allocator::MemoryAllocator,
    render_pass::Framebuffer,
    sync::GpuFuture,
};
use winit::event::WindowEvent;

use crate::{camera::Camera, renderer::Renderer};

pub mod logger;
pub mod camera_view;
pub mod scene_tree;
pub mod renderer_settings_edit;
pub mod edit_widgets;

pub enum Pane {
    CameraView(CameraView),
    SceneTree(SceneTree),
    LogView,
    RendererSettingsEdit(RendererSettingsEdit),
}

pub struct UserInterface {
    tree: Tree<Pane>,
    behavior: TreeBehavior,
}

impl UserInterface {
    pub fn new(
        memory_allocator: Arc<dyn MemoryAllocator>,
        renderer: Arc<RwLock<Renderer>>,
        gui: Gui,
        frames_in_flight: usize,
    ) -> Self {
        let mut tiles = egui_tiles::Tiles::default();
        let root = tiles.insert_tab_tile(vec![]);

        let tree = egui_tiles::Tree::new("my_tree", root, tiles);

        let behavior = TreeBehavior {
            memory_allocator,
            renderer,
            gui,
            image_index: 0,
            frames_in_flight,
            cameras_to_draw: vec![],
            focused_camera: None,
        };

        Self { tree, behavior }
    }

    pub fn update(&mut self, winit_event: &WindowEvent) -> bool {
        self.behavior.gui.update(winit_event)
    }

    pub fn build(
        &mut self,
        image_index: usize,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        frametime: Duration,
    ) {
        self.behavior.gui.begin_frame();
        let ctx = self.behavior.gui.context();
        egui::CentralPanel::default()
            .frame(Frame::default().inner_margin(Margin::same(0)))
            .show(&ctx, |ui| {
                self.behavior.image_index = image_index;
                ui.label(format!("FPS: {}", 1.0 / frametime.as_secs_f32()));
                self.tree.ui(&mut self.behavior, ui);
            });

        for (camera, framebuffer) in &self.behavior.cameras_to_draw {
            self.behavior
                .renderer
                .write()
                .unwrap()
                .draw(builder, &framebuffer, &camera);
        }
        self.behavior.cameras_to_draw.clear();
    }

    pub fn draw<F>(&mut self, previous_future: F, image_view: &Arc<ImageView>) -> Box<dyn GpuFuture>
    where
        F: GpuFuture + 'static,
    {
        self.behavior
            .gui
            .draw_on_image(previous_future, image_view.clone())
    }

    pub fn add_camera_view(&mut self, camera: Camera) {
        let camera_view = CameraView::new(camera);

        let id = self.tree.tiles.insert_pane(Pane::CameraView(camera_view));
        self.tree.root = Some({
            let old_root = self.tree.root().unwrap();
            let tiles = &mut self.tree.tiles;
            tiles.insert_tab_tile(vec![old_root, id])
        });
    }

    pub fn add_log_view(&mut self) {
        let id = self.tree.tiles.insert_pane(Pane::LogView);
        self.tree.root = Some({
            let old_root = self.tree.root().unwrap();
            let tiles = &mut self.tree.tiles;
            tiles.insert_tab_tile(vec![old_root, id])
        });
    }
    
    pub fn add_scene_tree(&mut self, world: World) {
        let id = self.tree.tiles.insert_pane(Pane::SceneTree(SceneTree::new(world)));
        self.tree.root = Some({
            let old_root = self.tree.root().unwrap();
            let tiles = &mut self.tree.tiles;
            tiles.insert_tab_tile(vec![old_root, id])
        });
    }
    
    pub fn add_renderer_settings_editor(&mut self, renderer: Arc<RwLock<Renderer>>) {
        let id = self.tree.tiles.insert_pane(Pane::RendererSettingsEdit(RendererSettingsEdit::new(renderer)));
        self.tree.root = Some({
            let old_root = self.tree.root().unwrap();
            let tiles = &mut self.tree.tiles;
            tiles.insert_tab_tile(vec![old_root, id])
        });
    }

    pub fn get_focused_camera(&mut self) -> Option<&mut Camera> {
        let tile = self.tree.tiles.get_mut(self.behavior.focused_camera?).unwrap();

        let pane = match tile {
            egui_tiles::Tile::Pane(pane) => pane,
            _ => panic!("Focused camera TileId is from a non Pane tile"),
        };

        match pane {
            Pane::CameraView(camera_view) => Some(camera_view.camera_mut()),
            _ => panic!("Focused camera TileId is from non camera pane"),
        }
    }

    pub fn unfocus_camera(&mut self) {
        self.behavior.focused_camera = None;
    }
}

pub struct TreeBehavior {
    memory_allocator: Arc<dyn MemoryAllocator>,
    renderer: Arc<RwLock<Renderer>>,
    gui: Gui,
    image_index: usize,
    frames_in_flight: usize,
    cameras_to_draw: Vec<(Camera, Arc<Framebuffer>)>,
    focused_camera: Option<egui_tiles::TileId>,
}

impl egui_tiles::Behavior<Pane> for TreeBehavior {
    fn tab_title_for_pane(&mut self, pane: &Pane) -> egui::WidgetText {
        match pane {
            Pane::CameraView(_) => "Camera view",
            Pane::SceneTree(_) => "Scene tree",
            Pane::LogView => "Log",
            Pane::RendererSettingsEdit(_) => "Renderer settings",
        }
        .into()
    }

    fn pane_ui(
        &mut self,
        ui: &mut egui::Ui,
        tile_id: egui_tiles::TileId,
        pane: &mut Pane,
    ) -> egui_tiles::UiResponse {
        let (started_dragging, pressed_close) = ui
            .horizontal(|ui| {
                (
                    ui.add(egui::Button::new("Drag me!").sense(egui::Sense::drag()))
                        .drag_started(),
                    ui.button("X").clicked(),
                )
            })
            .inner;

        if pressed_close {
            println!("CLOSED");
        }

        match pane {
            Pane::CameraView(camera_view) => {
                let clicked = camera_view.draw(
                    ui,
                    &mut self.gui,
                    &self.memory_allocator,
                    &self.renderer,
                    self.image_index,
                    self.frames_in_flight,
                    &mut self.cameras_to_draw,
                );

                if clicked {
                    self.focused_camera = Some(tile_id);
                }
            }
            Pane::SceneTree(scene_tree) => {
                scene_tree.draw(ui);
            }
            Pane::LogView => {
                logger::ui(ui);
            }
            Pane::RendererSettingsEdit(renderer_settings_edit) => {
                renderer_settings_edit.draw(ui);
            }
        }

        if started_dragging {
            egui_tiles::UiResponse::DragStarted
        } else {
            egui_tiles::UiResponse::None
        }
    }
}

