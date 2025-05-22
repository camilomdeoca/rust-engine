use std::sync::Arc;

use flecs_ecs::prelude::*;
use glam::{Mat3, Mat4, Vec3};
use sdl2::video::Window;
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        BufferUsage, Subbuffer,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferExecFuture, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
        SubpassBeginInfo, SubpassContents, SubpassEndInfo,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, layout::DescriptorType, DescriptorBufferInfo,
        DescriptorSet, WriteDescriptorSet,
    },
    device::{Device, DeviceOwned, Queue},
    format::Format,
    image::{
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, LOD_CLAMP_NONE},
        view::ImageView,
        Image, ImageCreateInfo, ImageType, ImageUsage,
    },
    instance::Instance,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    padded::Padded,
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{CompareOp, DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::{CullMode, RasterizationState},
            vertex_input::{Vertex as _, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::EntryPoint,
    swapchain::{
        acquire_next_image, PresentFuture, PresentMode, Surface, Swapchain, SwapchainAcquireFuture,
        SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{
        self,
        future::{FenceSignalFuture, JoinFuture},
        GpuFuture,
    },
    DeviceSize, Validated, VulkanError,
};

use crate::{
    assets::{
        loaders::{mesh_loader::load_mesh_from_buffers, texture_loader::load_texture_from_buffer},
        vertex::Vertex,
    },
    camera::Camera,
    ecs::components::{self, Material, Mesh, Transform},
};

const MAX_FRAMES_IN_FLIGHT: usize = 2;

struct DefaultMaterialTextures {
    pub diffuse: Arc<ImageView>,
    pub metallic_roughness: Arc<ImageView>,
    pub ambient_oclussion: Arc<ImageView>,
    pub emissive: Arc<ImageView>,
    pub normal: Arc<ImageView>,
}

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
    mesh_vs: EntryPoint,
    mesh_fs: EntryPoint,
    mesh_pipeline: Arc<GraphicsPipeline>,
    skybox_vs: EntryPoint,
    skybox_fs: EntryPoint,
    skybox_pipeline: Arc<GraphicsPipeline>,
    cube_vertex_buffer: Subbuffer<[Vertex]>,
    cube_index_buffer: Subbuffer<[u32]>,
    recreate_swapchain: bool,
    fences: [Option<
        Arc<
            FenceSignalFuture<
                PresentFuture<
                    CommandBufferExecFuture<JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture>>,
                >,
            >,
        >,
    >; MAX_FRAMES_IN_FLIGHT],
    previous_fence_index: u32,

    default_material_textures: DefaultMaterialTextures,
}

fn create_default_material_textures(
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
) -> DefaultMaterialTextures {
    let format = Format::R8G8B8A8_UNORM;
    let diffuse = load_texture_from_buffer(
        memory_allocator.clone(),
        command_buffer_allocator.clone(),
        queue.clone(),
        format,
        [1; 2],
        &[255, 0, 255, 255],
    )
    .unwrap();
    let metallic_roughness = load_texture_from_buffer(
        memory_allocator.clone(),
        command_buffer_allocator.clone(),
        queue.clone(),
        format,
        [1; 2],
        &[0, 255, 0, 255],
    )
    .unwrap();
    let ambient_oclussion = load_texture_from_buffer(
        memory_allocator.clone(),
        command_buffer_allocator.clone(),
        queue.clone(),
        Format::R8_UNORM,
        [1; 2],
        &[255; 1],
    )
    .unwrap();
    let emissive = load_texture_from_buffer(
        memory_allocator.clone(),
        command_buffer_allocator.clone(),
        queue.clone(),
        format,
        [1; 2],
        &[255; 4],
    )
    .unwrap();
    let normal = load_texture_from_buffer(
        memory_allocator.clone(),
        command_buffer_allocator.clone(),
        queue.clone(),
        format,
        [1; 2],
        &[0, 0, 255, 255],
    )
    .unwrap();
    DefaultMaterialTextures {
        diffuse,
        metallic_roughness,
        ambient_oclussion,
        emissive,
        normal,
    }
}

fn window_size_dependent_setup(
    window_size: [u32; 2],
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    mesh_vs: &EntryPoint,
    mesh_fs: &EntryPoint,
    skybox_vs: &EntryPoint,
    skybox_fs: &EntryPoint,
) -> (
    Vec<Arc<Framebuffer>>,
    (Arc<GraphicsPipeline>, Arc<GraphicsPipeline>),
) {
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

    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: window_size.map(|e| e as f32),
        depth_range: 0.0..=1.0,
    };

    let mesh_pipeline = {
        let mesh_pipeline_stages = [
            PipelineShaderStageCreateInfo::new(mesh_vs.clone()),
            PipelineShaderStageCreateInfo::new(mesh_fs.clone()),
        ];
        let mut layout_create_info =
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&mesh_pipeline_stages);

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                lod: 0.0..=LOD_CLAMP_NONE,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .unwrap();
        // Env descriptor set (rebound when the skybox changes, not too often)
        layout_create_info.set_layouts[0]
            .bindings
            .get_mut(&0)
            .unwrap()
            .immutable_samplers = vec![sampler];

        // Frame descriptor set (bound during the whole frame)
        layout_create_info.set_layouts[1]
            .bindings
            .get_mut(&0)
            .unwrap()
            .descriptor_type = DescriptorType::UniformBuffer;

        // Material descriptor set (rebound for each material)

        // Model descriptor set (rebound at an offset per model)
        layout_create_info.set_layouts[3]
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

        let mesh_pass = Subpass::from(render_pass.clone(), 0).unwrap();

        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: mesh_pipeline_stages.into_iter().collect(),
                vertex_input_state: Some(Vertex::per_vertex().definition(mesh_vs).unwrap()),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState {
                    viewports: [viewport.clone()].into_iter().collect(),
                    ..Default::default()
                }),
                rasterization_state: Some(RasterizationState {
                    cull_mode: CullMode::Back,
                    ..Default::default()
                }),
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState::simple()),
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    mesh_pass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(mesh_pass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    };

    let skybox_pipeline = {
        let stages = [
            PipelineShaderStageCreateInfo::new(skybox_vs.clone()),
            PipelineShaderStageCreateInfo::new(skybox_fs.clone()),
        ];
        let mut layout_create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .unwrap();
        // Env descriptor set (rebound when the skybox changes, not too often)
        layout_create_info.set_layouts[0]
            .bindings
            .get_mut(&0)
            .unwrap()
            .immutable_samplers = vec![sampler];

        // Frame descriptor set (bound during the whole frame)
        layout_create_info.set_layouts[1]
            .bindings
            .get_mut(&0)
            .unwrap()
            .descriptor_type = DescriptorType::UniformBuffer;

        let layout = PipelineLayout::new(
            device.clone(),
            layout_create_info
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let skybox_pass = Subpass::from(render_pass.clone(), 1).unwrap();

        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(Vertex::per_vertex().definition(skybox_vs).unwrap()),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState {
                    viewports: [viewport.clone()].into_iter().collect(),
                    ..Default::default()
                }),
                rasterization_state: Some(RasterizationState {
                    cull_mode: CullMode::None,
                    ..Default::default()
                }),
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState {
                        write_enable: false,
                        compare_op: CompareOp::LessOrEqual,
                    }),
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    skybox_pass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(skybox_pass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    };

    (framebuffers, (mesh_pipeline, skybox_pipeline))
}

impl Renderer {
    pub fn new(
        instance: Arc<Instance>,
        window: Arc<Window>,
        device: Arc<Device>,
        queue: Arc<Queue>,
        memory_allocator: Arc<StandardMemoryAllocator>,
    ) -> Self {
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

        let render_pass = vulkano::ordered_passes_renderpass!(
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
            passes: [
                // Mesh rendering
                {
                    color: [color],
                    depth_stencil: {depth_stencil},
                    input: []
                },
                // Skybox (after so it doesnt draw behind the meshes)
                {
                    color: [color],
                    depth_stencil: {depth_stencil},
                    input: []
                }
            ]
        )
        .unwrap();

        let mesh_vs = mesh_shaders::vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let mesh_fs = mesh_shaders::fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let skybox_vs = skybox_shaders::vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let skybox_fs = skybox_shaders::fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let (framebuffers, (mesh_pipeline, skybox_pipeline)) = window_size_dependent_setup(
            <[u32; 2]>::from(window_size),
            &images,
            &render_pass,
            &memory_allocator,
            &mesh_vs,
            &mesh_fs,
            &skybox_vs,
            &skybox_fs,
        );

        let (cube_vertex_buffer, cube_index_buffer) = load_mesh_from_buffers(
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            queue.clone(),
            [
                Vertex {
                    a_position: [-1.0, -1.0, 1.0],
                    a_normal: [0.0; 3],
                    a_tangent: [0.0; 3],
                    a_uv: [0.0; 2],
                },
                Vertex {
                    a_position: [1.0, -1.0, 1.0],
                    a_normal: [0.0; 3],
                    a_tangent: [0.0; 3],
                    a_uv: [0.0; 2],
                },
                Vertex {
                    a_position: [-1.0, 1.0, 1.0],
                    a_normal: [0.0; 3],
                    a_tangent: [0.0; 3],
                    a_uv: [0.0; 2],
                },
                Vertex {
                    a_position: [1.0, 1.0, 1.0],
                    a_normal: [0.0; 3],
                    a_tangent: [0.0; 3],
                    a_uv: [0.0; 2],
                },
                Vertex {
                    a_position: [-1.0, -1.0, -1.0],
                    a_normal: [0.0; 3],
                    a_tangent: [0.0; 3],
                    a_uv: [0.0; 2],
                },
                Vertex {
                    a_position: [1.0, -1.0, -1.0],
                    a_normal: [0.0; 3],
                    a_tangent: [0.0; 3],
                    a_uv: [0.0; 2],
                },
                Vertex {
                    a_position: [-1.0, 1.0, -1.0],
                    a_normal: [0.0; 3],
                    a_tangent: [0.0; 3],
                    a_uv: [0.0; 2],
                },
                Vertex {
                    a_position: [1.0, 1.0, -1.0],
                    a_normal: [0.0; 3],
                    a_tangent: [0.0; 3],
                    a_uv: [0.0; 2],
                },
            ],
            [
                // Front face (inverted)
                2, 1, 0, 3, 1, 2, // Back face (inverted)
                5, 6, 4, 7, 6, 5, // Left face (inverted)
                4, 2, 0, 6, 2, 4, // Right face (inverted)
                3, 5, 1, 7, 5, 3, // Top face (inverted)
                6, 3, 2, 7, 3, 6, // Bottom face (inverted)
                1, 4, 0, 5, 4, 1,
            ],
        )
        .unwrap();

        let default_material_textures = create_default_material_textures(
            queue.clone(), 
            memory_allocator.clone(), 
            command_buffer_allocator.clone(),
        );

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
            mesh_vs,
            mesh_fs,
            skybox_vs,
            skybox_fs,
            mesh_pipeline,
            skybox_pipeline,
            cube_vertex_buffer,
            cube_index_buffer,
            fences: [const { None }; MAX_FRAMES_IN_FLIGHT],
            recreate_swapchain: false,
            previous_fence_index: 0,
            default_material_textures,
        }
    }

    pub fn draw(&mut self, camera: &Camera, world: &World) {
        let window_size = self.window.size();
        if window_size.0 == 0 || window_size.1 == 0 {
            return;
        }

        if self.recreate_swapchain {
            let (new_swapchain, new_images) = self
                .swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent: window_size.into(),
                    ..self.swapchain.create_info()
                })
                .expect("Failed to recreate swapchain");

            self.swapchain = new_swapchain;

            (
                self.framebuffers,
                (self.mesh_pipeline, self.skybox_pipeline),
            ) = window_size_dependent_setup(
                window_size.into(),
                &new_images,
                &self.render_pass,
                &self.memory_allocator,
                &self.mesh_vs,
                &self.mesh_fs,
                &self.skybox_vs,
                &self.skybox_fs,
            );

            self.recreate_swapchain = false;
        }

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(self.swapchain.clone(), None).map_err(Validated::unwrap) {
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

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into()), Some(1f32.into())],

                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[image_index as usize].clone(),
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap();

        self.draw_meshes(&mut builder, &camera, &world);

        builder
            .next_subpass(
                SubpassEndInfo::default(),
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap();

        self.draw_skybox(&mut builder, &camera, &world);

        builder.end_render_pass(Default::default()).unwrap();

        let command_buffer = builder.build().unwrap();

        let future = previous_future
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index),
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

    fn draw_meshes(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        camera: &Camera,
        world: &World,
    ) {
        let mut irradiance_map = None;
        let mut prefiltered_environment_map = None;
        let mut environment_brdf_lut = None;
        world
            .query::<&components::EnvironmentCubemap>()
            .term_at(0)
            .singleton()
            .build()
            .each(|env_cubemap| {
                irradiance_map = Some(env_cubemap.irradiance_map.clone());
                prefiltered_environment_map = Some(env_cubemap.prefiltered_environment_map.clone());
                environment_brdf_lut = Some(env_cubemap.environment_brdf_lut.clone());
            });
        let irradiance_map = irradiance_map.unwrap();
        let prefiltered_environment_map = prefiltered_environment_map.unwrap();
        let environment_brdf_lut = environment_brdf_lut.unwrap();

        let environment_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.mesh_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view(1, irradiance_map.clone()),
                WriteDescriptorSet::image_view(2, prefiltered_environment_map.clone()),
                WriteDescriptorSet::image_view(3, environment_brdf_lut.clone()),
            ],
            [],
        )
        .unwrap();

        let frame_uniform_buffer = {
            let aspect_ratio =
                self.swapchain.image_extent()[0] as f32 / self.swapchain.image_extent()[1] as f32;

            let proj = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0))
                * Mat4::perspective_rh_gl(camera.fov, aspect_ratio, 0.01, 100.0);
            let view = Mat4::look_to_rh(
                camera.position,
                camera.rotation * Vec3::NEG_Z,
                camera.rotation * Vec3::Y,
            );

            let uniform_data = mesh_shaders::vs::FrameUniforms {
                view: view.to_cols_array_2d(),
                inv_view: Mat3::from_mat4(view)
                    .inverse()
                    .to_cols_array_2d()
                    .map(Padded),
                proj: proj.to_cols_array_2d(),
            };

            let buffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
            *buffer.write().unwrap() = uniform_data;

            buffer
        };

        let frame_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.mesh_pipeline.layout().set_layouts()[1].clone(),
            [WriteDescriptorSet::buffer(0, frame_uniform_buffer)],
            [],
        )
        .unwrap();

        let model_uniform_buffer = self
            .uniform_buffer_allocator
            .allocate_slice(16 * 1024)
            .unwrap();

        builder
            .bind_pipeline_graphics(self.mesh_pipeline.clone())
            .unwrap();

        let min_dynamic_align = self
            .device
            .physical_device()
            .properties()
            .min_uniform_buffer_offset_alignment
            .as_devicesize();

        let align =
            (size_of::<mesh_shaders::vs::ModelUniforms>() as DeviceSize + min_dynamic_align - 1)
                & !(min_dynamic_align - 1);

        let mut models_unforms = Vec::with_capacity(100); // maybe save it in the renderer so it
                                                          // isnt allocated every frame
        world.each::<(
            &Mesh,
            Option<&components::Material>,
            &Transform,
        )>(|(mesh, material, transform)| {
            let model_matrix = Mat4::from_scale_rotation_translation(
                transform.scale,
                transform.rotation,
                transform.translation,
            );

            let model_index = models_unforms.len() as DeviceSize;
            models_unforms.push(mesh_shaders::vs::ModelUniforms {
                transform: model_matrix.to_cols_array_2d(),
            });

            let model_descriptor_set = DescriptorSet::new(
                self.descriptor_set_allocator.clone(),
                self.mesh_pipeline.layout().set_layouts()[3].clone(),
                [WriteDescriptorSet::buffer_with_range(
                    0,
                    DescriptorBufferInfo {
                        buffer: model_uniform_buffer.clone(),
                        range: 0..size_of::<mesh_shaders::vs::ModelUniforms>() as DeviceSize,
                    },
                )],
                [],
            )
            .unwrap();

            let material = material.cloned().unwrap_or(Material::default());

            let diffuse = material.diffuse
                .unwrap_or(self.default_material_textures.diffuse.clone());
            let metallic_roughness = material.metallic_roughness
                .unwrap_or(self.default_material_textures.metallic_roughness.clone());
            let ambient_oclussion = material.ambient_oclussion
                .unwrap_or(self.default_material_textures.ambient_oclussion.clone());
            let emissive = material.emissive
                .unwrap_or(self.default_material_textures.emissive.clone());
            let normal = material.normal
                .unwrap_or(self.default_material_textures.normal.clone());

            let material_descriptor_set = DescriptorSet::new(
                self.descriptor_set_allocator.clone(),
                self.mesh_pipeline.layout().set_layouts()[2].clone(),
                [
                    WriteDescriptorSet::image_view(0, diffuse),
                    WriteDescriptorSet::image_view(1, metallic_roughness),
                    WriteDescriptorSet::image_view(2, ambient_oclussion),
                    WriteDescriptorSet::image_view(3, emissive),
                    WriteDescriptorSet::image_view(4, normal),
                ],
                [],
            )
            .unwrap();

            let offset = (model_index * align) as u32;
            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.mesh_pipeline.layout().clone(),
                    0,
                    (
                        environment_descriptor_set.clone(),
                        frame_descriptor_set.clone(),
                        material_descriptor_set.clone(),
                        model_descriptor_set.clone().offsets([offset]),
                    ),
                )
                .unwrap()
                .bind_vertex_buffers(0, mesh.vertex_buffer.clone())
                .unwrap()
                .bind_index_buffer(mesh.index_buffer.clone())
                .unwrap();

            unsafe { builder.draw_indexed(mesh.index_buffer.len() as u32, 1, 0, 0, 0) }.unwrap();
        });

        for (model_index, model_uniforms) in models_unforms.iter().enumerate() {
            *model_uniform_buffer
                .clone()
                .slice(model_index as DeviceSize * align..)
                .cast_aligned()
                .index(0)
                .write()
                .unwrap() = model_uniforms.clone();
        }
    }

    fn draw_skybox(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        camera: &Camera,
        world: &World,
    ) {
        builder
            .bind_pipeline_graphics(self.skybox_pipeline.clone())
            .unwrap();

        let mut environment_map = None;
        world
            .query::<&components::EnvironmentCubemap>()
            .term_at(0)
            .singleton()
            .build()
            .each(|env_cubemap| {
                environment_map = Some(env_cubemap.environment_map.clone());
            });
        let environment_map = environment_map.unwrap();

        let environment_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.skybox_pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::image_view(1, environment_map.clone())],
            [],
        )
        .unwrap();

        let frame_uniform_buffer = {
            let aspect_ratio =
                self.swapchain.image_extent()[0] as f32 / self.swapchain.image_extent()[1] as f32;

            let proj = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0)) // Correction for vulkan's
                                                                   // inverted y
                * Mat4::perspective_rh_gl(camera.fov, aspect_ratio, 0.01, 100.0);
            let view = Mat4::look_at_rh(
                Vec3::ZERO,
                camera.rotation * Vec3::NEG_Z,
                camera.rotation * Vec3::Y,
            );

            let uniform_data = skybox_shaders::vs::FrameUniforms {
                view_proj: (proj * view).to_cols_array_2d(),
            };

            let buffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
            *buffer.write().unwrap() = uniform_data;

            buffer
        };

        let frame_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.skybox_pipeline.layout().set_layouts()[1].clone(),
            [WriteDescriptorSet::buffer(0, frame_uniform_buffer)],
            [],
        )
        .unwrap();

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.skybox_pipeline.layout().clone(),
                0,
                (
                    environment_descriptor_set.clone(),
                    frame_descriptor_set.clone(),
                ),
            )
            .unwrap()
            .bind_vertex_buffers(0, self.cube_vertex_buffer.clone())
            .unwrap()
            .bind_index_buffer(self.cube_index_buffer.clone())
            .unwrap();

        unsafe { builder.draw_indexed(self.cube_index_buffer.len() as u32, 1, 0, 0, 0) }.unwrap();
    }
}

mod mesh_shaders {
    pub mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "assets/shaders/vertex.glsl",
        }
    }

    pub mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "assets/shaders/fragment.glsl",
        }
    }
}

mod skybox_shaders {
    pub mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "assets/shaders/skybox_vertex.glsl",
        }
    }

    pub mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "assets/shaders/skybox_fragment.glsl",
        }
    }
}
