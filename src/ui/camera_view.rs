use std::sync::{Arc, RwLock};

use egui::{load::SizedTexture, ImageSource, Sense, Ui};
use egui_winit_vulkano::Gui;
use glam::UVec2;
use vulkano::{
    format::Format,
    image::{
        sampler::SamplerCreateInfo, view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage,
    },
    memory::allocator::{AllocationCreateInfo, MemoryAllocator},
    render_pass::Framebuffer,
};

use crate::{camera::Camera, renderer::Renderer};

pub struct CameraView {
    camera: Camera,
    image_views: Vec<Arc<ImageView>>,
    egui_texture_ids: Vec<egui::TextureId>,
    framebuffers: Vec<Arc<Framebuffer>>,
}

impl CameraView {
    pub fn new(camera: Camera) -> Self {
        Self {
            camera,
            image_views: vec![],
            egui_texture_ids: vec![],
            framebuffers: vec![],
        }
    }

    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    pub fn draw(
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
        let response = ui
            .image(ImageSource::Texture(SizedTexture::new(
                self.egui_texture_ids[image_index as usize].clone(),
                (
                    self.image_views[image_index].image().extent()[0] as f32,
                    self.image_views[image_index].image().extent()[1] as f32,
                ),
            )))
            .interact(Sense::click());
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

        self.framebuffers = renderer
            .write()
            .unwrap()
            .resize_and_create_framebuffers(&self.image_views)
            .unwrap();

        self.egui_texture_ids = self
            .image_views
            .iter()
            .map(|image_view| {
                gui.register_user_image_view(image_view.clone(), SamplerCreateInfo::default())
            })
            .collect();
    }
}
