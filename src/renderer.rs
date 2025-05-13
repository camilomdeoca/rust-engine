use std::sync::Arc;

use flecs_ecs::prelude::*;
use glam::{Mat4, Vec3};
use sdl2::video::Window;
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        BufferUsage, Subbuffer,
    }, command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents
    }, descriptor_set::{
        allocator::{DescriptorSetAllocator, StandardDescriptorSetAllocator}, layout::DescriptorType, DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet
    }, device::{Device, DeviceOwned, Queue}, format::Format, image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage}, instance::Instance, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex as _, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    }, render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass}, shader::EntryPoint, swapchain::{
        acquire_next_image, PresentFuture, PresentMode, Surface, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo, SwapchainPresentInfo
    }, sync::{self, future::{FenceSignalFuture, JoinFuture}, GpuFuture}, DeviceSize, Validated, VulkanError
};

use crate::{
    assets::{database::AssetDatabase, Mesh, Vertex},
    camera::Camera, ecs::components,
};

const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct Renderer {
    window: Arc<Window>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    uniform_buffer_allocator: SubbufferAllocator,
    render_pass: Arc<RenderPass>,
    swapchain: Arc<Swapchain>,
    framebuffers: Vec<Arc<Framebuffer>>,
    vs: EntryPoint,
    fs: EntryPoint,
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
    recreate_swapchain: bool,
    fences: [Option<Arc<FenceSignalFuture<PresentFuture<CommandBufferExecFuture<JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture>>>>>>; MAX_FRAMES_IN_FLIGHT],
    previous_fence_index: u32,
}

fn window_size_dependent_setup(
    window_size: [u32; 2],
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    vs: &EntryPoint,
    fs: &EntryPoint,
) -> (Vec<Arc<Framebuffer>>, Arc<GraphicsPipeline>) {
    let device = memory_allocator.device();

    let depth_buffer = ImageView::new_default(
        Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::D16_UNORM,
                extent: images[0].extent(),
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap(),
    )
    .unwrap();

    let framebuffers = images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();

            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    let pipeline = {
        let vertex_input_state = Vertex::per_vertex()
            .definition(vs)
            .unwrap();
        let stages = [
            PipelineShaderStageCreateInfo::new(vs.clone()),
            PipelineShaderStageCreateInfo::new(fs.clone()),
        ];
        let mut layout_create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
        layout_create_info
            .set_layouts[0]
            .bindings
            .get_mut(&0)
            .unwrap()
            .descriptor_type = DescriptorType::UniformBuffer;
        layout_create_info
            .set_layouts[1]
            .bindings
            .get_mut(&0)
            .unwrap()
            .descriptor_type = DescriptorType::UniformBufferDynamic;
        let layout = PipelineLayout::new(
            device.clone(),
            layout_create_info
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState {
                    viewports: [Viewport {
                        offset: [0.0, 0.0],
                        extent: window_size.map(|e| e as f32),
                        depth_range: 0.0..=1.0,
                    }]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                }),
                rasterization_state: Some(RasterizationState::default()),
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState::simple()),
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    };

    (framebuffers, pipeline)
}

impl Renderer {
    pub fn new(instance: Arc<Instance>, window: Arc<Window>, device: Arc<Device>, queue: Arc<Queue>, memory_allocator: Arc<StandardMemoryAllocator>) -> Self {
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let uniform_buffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let surface = unsafe { Surface::from_window_ref(instance.clone(), &window).unwrap() };
        let window_size = window.size();

        let (swapchain, images) = {
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            let (image_format, _) = device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0];

            Swapchain::new(
                device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    present_mode: PresentMode::Fifo,
                    image_extent: window_size.into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
                depth_stencil: {
                    format: Format::D16_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {depth_stencil},
            },
        )
        .unwrap();

        let vs = vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let (framebuffers, pipeline) = window_size_dependent_setup(
            <[u32; 2]>::from(window_size),
            &images,
            &render_pass,
            &memory_allocator,
            &vs,
            &fs,
        );

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [window_size.0 as f32, window_size.1 as f32],
            depth_range: 0.0..=1.0,
        };

        Self {
            window,
            device,
            queue,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            uniform_buffer_allocator,
            render_pass,
            swapchain,
            framebuffers,
            vs,
            fs,
            pipeline,
            viewport,
            fences: [const { None }; MAX_FRAMES_IN_FLIGHT],
            recreate_swapchain: false,
            previous_fence_index: 0,
        }
    }

    pub fn draw(&mut self, camera: &Camera, world: &World, asset_database: &AssetDatabase) {
        let window_size = self.window.size();
        if window_size.0 == 0 || window_size.1 == 0 {
            return;
        }

        if self.recreate_swapchain {
            let (new_swapchain, new_images) = self.swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent: window_size.into(),
                    ..self.swapchain.create_info()
                })
                .expect("Failed to recreate swapchain");

            self.swapchain = new_swapchain;

            (self.framebuffers, self.pipeline) = window_size_dependent_setup(
                window_size.into(),
                &new_images,
                &self.render_pass,
                &self.memory_allocator,
                &self.vs,
                &self.fs,
            );

            self.viewport.extent = [window_size.0 as f32, window_size.1 as f32];
            self.recreate_swapchain = false;
        }

        let camera_uniform_buffer = {
            let aspect_ratio = self.swapchain.image_extent()[0] as f32
                / self.swapchain.image_extent()[1] as f32;

            let proj = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0)) * Mat4::perspective_rh_gl(
                camera.fov,
                aspect_ratio,
                0.01,
                100.0,
            );
            let view = Mat4::look_to_rh(
                camera.position,
                camera.rotation * Vec3::NEG_Z,
                camera.rotation * Vec3::Y
            );

            let uniform_data = vs::CameraUniforms {
                world: Mat4::IDENTITY.to_cols_array_2d(),
                view: view.to_cols_array_2d(),
                proj: proj.to_cols_array_2d(),
            };

            let buffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
            *buffer.write().unwrap() = uniform_data;

            buffer
        };

        let camera_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::buffer(0, camera_uniform_buffer)],
            [],
        )
        .unwrap();

        let (image_index, suboptimal, acquire_future) = match acquire_next_image(
            self.swapchain.clone(),
            None,
        )
        .map_err(Validated::unwrap) {
            Ok(r) => r,
            Err(VulkanError::OutOfDate) => {
                self.recreate_swapchain = true;
                return;
            }
            Err(e) => panic!("failed to acquire next image: {e}"),
        };

        if suboptimal {
            self.recreate_swapchain = true;
        }

        if let Some(image_fence) = &self.fences[image_index as usize] {
            image_fence.wait(None).unwrap();
        }

        let previous_future = match self.fences[self.previous_fence_index as usize].clone() {
            // Create a NowFuture
            None => {
                let mut now = sync::now(self.device.clone());
                now.cleanup_finished();

                now.boxed()
            }
            // Use the existing FenceSignalFuture
            Some(fence) => fence.boxed(),
        };

        let model_uniform_buffer = self.uniform_buffer_allocator.allocate_slice(256).unwrap();

        let model_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[1].clone(),
            [WriteDescriptorSet::buffer(0, model_uniform_buffer.clone())],
            [],
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some([0.0, 0.0, 1.0, 1.0].into()),
                        Some(1f32.into()),
                    ],

                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[image_index as usize].clone(),
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap()
            .set_viewport(0, [self.viewport.clone()].into_iter().collect())
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap();

        let model_index = 0;
        world.each::<(&components::Mesh, &components::Transform)>(|(mesh, transform)| {
            let mesh = asset_database.get_mesh(&mesh.id).unwrap();
            let model_matrix = Mat4::from_scale_rotation_translation(
                transform.scale,
                transform.rotation,
                transform.translation
            );

            *model_uniform_buffer.clone().index(model_index).write().unwrap() = vs::ModelUniforms {
                transform: model_matrix.to_cols_array_2d()
            };

            let offset = (model_index * size_of::<vs::ModelUniforms>() as DeviceSize) as u32;
            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    0,
                    (camera_descriptor_set.clone(), model_descriptor_set.clone().offsets([offset]))
                )
                .unwrap()
                .bind_vertex_buffers(0, mesh.vertex_buffer.clone())
                .unwrap()
                .bind_index_buffer(mesh.index_buffer.clone())
                .unwrap();

            unsafe { builder.draw_indexed(mesh.index_buffer.len() as u32, 1, 0, 0, 0) }.unwrap();
        });

        builder
            .end_render_pass(Default::default())
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = previous_future
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    self.swapchain.clone(),
                    image_index,
                ),
            )
            .then_signal_fence_and_flush();

        self.fences[image_index as usize] = match future.map_err(Validated::unwrap) {
            Ok(value) => Some(Arc::new(value)),
            Err(VulkanError::OutOfDate) => {
                self.recreate_swapchain = true;
                None
            }
            Err(e) => {
                println!("failed to flush future: {e}");
                None
            }
        };

        self.previous_fence_index = image_index;
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "assets/shaders/vertex.glsl",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "assets/shaders/fragment.glsl",
    }
}
