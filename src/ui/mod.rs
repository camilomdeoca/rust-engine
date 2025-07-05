use std::{sync::{Arc, RwLock}, time::Duration};

use egui::{load::SizedTexture, Frame, ImageSource, Margin, Sense, Ui};
use egui_tiles::Tree;
use egui_winit_vulkano::Gui;
use glam::UVec2;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    format::Format,
    image::{
        sampler::SamplerCreateInfo, view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage,
    },
    memory::allocator::{AllocationCreateInfo, MemoryAllocator},
    render_pass::Framebuffer,
    sync::GpuFuture,
};
use winit::event::WindowEvent;

use crate::{camera::Camera, renderer::Renderer};

pub mod logger;

struct CameraView {
    camera: Camera,
    image_views: Vec<Arc<ImageView>>,
    egui_texture_ids: Vec<egui::TextureId>,
    framebuffers: Vec<Arc<Framebuffer>>,
}

pub enum Pane {
    CameraView(CameraView),
    LogView,
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
        let camera_view = CameraView {
            camera,
            image_views: vec![],
            egui_texture_ids: vec![],
            framebuffers: vec![],
        };

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

    pub fn get_focused_camera(&mut self) -> Option<&mut Camera> {
        let tile = self.tree.tiles.get_mut(self.behavior.focused_camera?).unwrap();

        let pane = match tile {
            egui_tiles::Tile::Pane(pane) => pane,
            _ => panic!("Focused camera TileId is from a non Pane tile"),
        };

        match pane {
            Pane::CameraView(camera_view) => Some(&mut camera_view.camera),
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
            Pane::LogView => "Log",
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
            Pane::LogView => {
                logger::ui(ui);
            }
        }

        if started_dragging {
            egui_tiles::UiResponse::DragStarted
        } else {
            egui_tiles::UiResponse::None
        }
    }
}

impl CameraView {
    fn draw(
        &mut self,
        ui: &mut Ui,
        gui: &mut Gui,
        memory_allocator: &Arc<dyn MemoryAllocator>,
        renderer: &Arc<RwLock<Renderer>>,
        image_index: usize,
        frames_in_flight: usize,
        cameras_to_draw: &mut Vec<(Camera, Arc<Framebuffer>)>,
    ) -> bool {
        let available_size = ui.available_size();
        let size_changed = self.image_views.len() > 0
            && (available_size.x as u32 != self.image_views[0].image().extent()[0]
                || available_size.y as u32 != self.image_views[0].image().extent()[1]);
        // Resize renderer things (pipeline, etc) if size of widget changed
        // TODO: Check the index of the last frame it resized and if 10 frames or
        // so havent passed dont resize until then. Because its crashing :).
        // Vulkano issues page has something similar i dont know what is
        if size_changed || self.egui_texture_ids.is_empty() {
            self.resize(
                UVec2::new(available_size.x as u32, available_size.y as u32),
                gui,
                memory_allocator,
                renderer,
                frames_in_flight,
            );
        }

        // Render the renderer framebuffer
        let response = ui.image(ImageSource::Texture(SizedTexture::new(
            self.egui_texture_ids[image_index as usize].clone(),
            (
                self.image_views[image_index].image().extent()[0] as f32,
                self.image_views[image_index].image().extent()[1] as f32,
            ),
        ))).interact(Sense::click());
        cameras_to_draw.push((self.camera.clone(), self.framebuffers[image_index].clone()));

        return response.clicked();
    }

    fn resize(
        &mut self,
        size: UVec2,
        gui: &mut Gui,
        memory_allocator: &Arc<dyn MemoryAllocator>,
        renderer: &Arc<RwLock<Renderer>>,
        frames_in_flight: usize,
    ) {
        self.egui_texture_ids
            .iter()
            .for_each(|id| gui.unregister_user_image(id.clone()));

        // 3d Renderer doesnt render directly to the swapchain, it renders to a
        // texture that later egui renders in a widget in the screen
        self.image_views.clear();
        for _ in 0..frames_in_flight {
            self.image_views.push(
                ImageView::new_default(
                    Image::new(
                        memory_allocator.clone(),
                        ImageCreateInfo {
                            image_type: ImageType::Dim2d,
                            format: Format::R8G8B8A8_UNORM,
                            extent: [size.x, size.y, 1],
                            usage: ImageUsage::TRANSFER_DST
                                | ImageUsage::SAMPLED
                                | ImageUsage::COLOR_ATTACHMENT,
                            ..Default::default()
                        },
                        AllocationCreateInfo::default(),
                    )
                    .unwrap(),
                )
                .unwrap(),
            );
        }

        self.framebuffers = renderer.write().unwrap().resize_and_create_framebuffers(&self.image_views);

        self.egui_texture_ids = self
            .image_views
            .iter()
            .map(|image_view| {
                gui.register_user_image_view(image_view.clone(), SamplerCreateInfo::default())
            })
            .collect();
    }
}
