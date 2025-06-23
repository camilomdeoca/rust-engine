use std::{
    sync::{Arc, RwLock},
    u32,
};

use flecs_ecs::prelude::*;
use glam::{Mat3, Mat4, Vec3};
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        Buffer, BufferCreateInfo, BufferUsage, Subbuffer,
    }, command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        DrawIndexedIndirectCommand, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
        SubpassBeginInfo, SubpassContents, SubpassEndInfo,
    }, descriptor_set::{
        allocator::StandardDescriptorSetAllocator,
        layout::{
            DescriptorBindingFlags, DescriptorType,
        },
        DescriptorSet, WriteDescriptorSet,
    }, device::{Device, DeviceOwned, Queue}, format::Format, image::{
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, LOD_CLAMP_NONE},
        view::ImageView,
        Image, ImageCreateInfo, ImageType, ImageUsage,
    }, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, padded::Padded, pipeline::{
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
    }, render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass}, shader::EntryPoint, DeviceSize
};

use crate::{
    assets::{
        database::{AssetDatabase, AssetDatabaseChangeObserver},
        loaders::mesh_loader::load_mesh_from_buffers_into_new_buffers,
        vertex::Vertex,
    },
    camera::Camera,
    ecs::components::{self, MaterialComponent, MeshComponent, Transform},
    profile::ProfileTimer,
};

pub struct Renderer {
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    uniform_buffer_allocator: SubbufferAllocator,
    storage_buffer_allocator: SubbufferAllocator,
    indirect_buffer_allocator: SubbufferAllocator,

    asset_database: Arc<RwLock<AssetDatabase>>,

    render_pass: Arc<RenderPass>,
    mesh_vs: EntryPoint,
    mesh_fs: EntryPoint,
    mesh_pipeline: Arc<GraphicsPipeline>,
    skybox_vs: EntryPoint,
    skybox_fs: EntryPoint,
    skybox_pipeline: Arc<GraphicsPipeline>,
    cube_vertex_buffer: Subbuffer<[Vertex]>,
    cube_index_buffer: Subbuffer<[u32]>,

    materials_storage_buffer: Subbuffer<[mesh_shaders::fs::Material]>,

    sampler: Arc<Sampler>,
    environment_descriptor_set: Arc<DescriptorSet>,
    materials_descriptor_set: Arc<DescriptorSet>,

    environment_cubemap_query: Query<&'static components::EnvironmentCubemap>,
    meshes_with_materials_query: Query<(
        &'static MeshComponent,
        &'static MaterialComponent,
        &'static Transform,
    )>,
    world: World,
}

fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    mesh_vs: &EntryPoint,
    mesh_fs: &EntryPoint,
    skybox_vs: &EntryPoint,
    skybox_fs: &EntryPoint,
    sampler: &Arc<Sampler>,
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
        extent: [images[0].extent()[0] as f32, images[0].extent()[1] as f32],
        depth_range: 0.0..=1.0,
    };

    let properties = device.physical_device().properties();

    let mesh_pipeline = {
        let mesh_pipeline_stages = [
            PipelineShaderStageCreateInfo::new(mesh_vs.clone()),
            PipelineShaderStageCreateInfo::new(mesh_fs.clone()),
        ];
        let mut layout_create_info =
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&mesh_pipeline_stages);
        
        // Env descriptor set (rebound when the skybox changes, not too often)
        layout_create_info.set_layouts[3]
            .bindings
            .get_mut(&0)
            .unwrap()
            .immutable_samplers = vec![sampler.clone()];
        layout_create_info.set_layouts[3]
            .bindings
            .get_mut(&4)
            .unwrap()
            .descriptor_count = properties.max_descriptor_set_samplers - 3;
        layout_create_info.set_layouts[3]
            .bindings
            .get_mut(&4)
            .unwrap()
            .binding_flags = DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT | DescriptorBindingFlags::PARTIALLY_BOUND;

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

        // Env descriptor set (rebound when the skybox changes, not too often)
        layout_create_info.set_layouts[0]
            .bindings
            .get_mut(&0)
            .unwrap()
            .immutable_samplers = vec![sampler.clone()];

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

impl AssetDatabaseChangeObserver for Renderer {}

///
impl Renderer {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        asset_database: Arc<RwLock<AssetDatabase>>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        world: World,
        images: &[Arc<Image>],
    ) -> (Self, Vec<Arc<Framebuffer>>) {
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

        let storage_buffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::STORAGE_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let indirect_buffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::INDIRECT_BUFFER | BufferUsage::STORAGE_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let render_pass = vulkano::ordered_passes_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: images[0].format(),
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

        let (framebuffers, (mesh_pipeline, skybox_pipeline)) = window_size_dependent_setup(
            &images,
            &render_pass,
            &memory_allocator,
            &mesh_vs,
            &mesh_fs,
            &skybox_vs,
            &skybox_fs,
            &sampler,
        );

        let (cube_vertex_buffer, cube_index_buffer) = load_mesh_from_buffers_into_new_buffers(
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            &queue,
            vec![
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
            vec![
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

        let environment_cubemap_query = world
            .query::<&components::EnvironmentCubemap>()
            .term_at(0)
            .singleton()
            .build();

        let meshes_with_materials_query =
            world.new_query::<(&MeshComponent, &MaterialComponent, &Transform)>();

        let mut irradiance_map = None;
        let mut prefiltered_environment_map = None;
        let mut environment_brdf_lut = None;

        environment_cubemap_query.each(|env_cubemap| {
            irradiance_map = Some(env_cubemap.irradiance_map.clone());
            prefiltered_environment_map = Some(env_cubemap.prefiltered_environment_map.clone());
            environment_brdf_lut = Some(env_cubemap.environment_brdf_lut.clone());
        });
        let irradiance_map = irradiance_map.unwrap();
        let prefiltered_environment_map = prefiltered_environment_map.unwrap();
        let environment_brdf_lut = environment_brdf_lut.unwrap();

        let asset_database_read = asset_database.read().unwrap();
        println!("TEXTURE COUNT {}", asset_database_read.textures().len());
        let environment_descriptor_set = DescriptorSet::new_variable(
            descriptor_set_allocator.clone(),
            mesh_pipeline.layout().set_layouts()[3].clone(),
            asset_database_read.textures().len() as u32,
            [
                WriteDescriptorSet::image_view(
                    1,
                    asset_database_read
                        .get_cubemap(irradiance_map)
                        .unwrap()
                        .cubemap
                        .clone(),
                ),
                WriteDescriptorSet::image_view(
                    2,
                    asset_database_read
                        .get_cubemap(prefiltered_environment_map)
                        .unwrap()
                        .cubemap
                        .clone(),
                ),
                WriteDescriptorSet::image_view(
                    3,
                    asset_database_read
                        .get_cubemap(environment_brdf_lut)
                        .unwrap()
                        .cubemap
                        .clone(),
                ),
                WriteDescriptorSet::image_view_array(
                    4,
                    0,
                    asset_database_read
                        .textures()
                        .iter()
                        .map(|texture| texture.texture.clone()),
                ),
            ],
            [],
        )
        .unwrap();

        println!(
            "sizeof MATERIAL {}",
            size_of::<mesh_shaders::fs::Material>()
        );
        assert!(size_of::<mesh_shaders::fs::Material>() % 16 == 0);

        let materials_iter = asset_database_read.materials().iter().map(|material| {
            let diffuse = material.diffuse.clone().map(|id| id.0);
            let metallic_roughness = material.metallic_roughness.clone().map(|id| id.0);
            let ambient_oclussion = material.ambient_oclussion.clone().map(|id| id.0);
            let emissive = material.emissive.clone().map(|id| id.0);
            let normal = material.normal.clone().map(|id| id.0);
            mesh_shaders::fs::Material {
                base_color_factor: material.color_factor.into(),
                emissive_factor: material.emissive_factor.into(),
                metallic_factor: material.metallic_factor.into(),
                roughness_factor: material.roughness_factor.into(),
                base_color_texture_id: diffuse.unwrap_or(u32::MAX),
                metallic_roughness_texture_id: metallic_roughness.unwrap_or(u32::MAX),
                ambient_oclussion_texture_id: ambient_oclussion.unwrap_or(u32::MAX).into(),
                emissive_texture_id: emissive.unwrap_or(u32::MAX),
                normal_texture_id: normal.unwrap_or(u32::MAX),
                pad: [0; 3],
            }
        });

        let materials_storage_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            materials_iter,
        )
        .unwrap();
        drop(asset_database_read);

        let materials_descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            mesh_pipeline.layout().set_layouts()[1].clone(),
            [WriteDescriptorSet::buffer(
                0,
                materials_storage_buffer.clone(),
            )],
            [],
        )
        .unwrap();

        println!(
            "VARIABLE_DESCRIPTOR_COUNT: {:#?}",
            environment_descriptor_set.variable_descriptor_count()
        );

        let renderer = Self {
            memory_allocator,
            descriptor_set_allocator,
            uniform_buffer_allocator,
            storage_buffer_allocator,
            indirect_buffer_allocator,
            asset_database,
            render_pass,
            mesh_vs,
            mesh_fs,
            skybox_vs,
            skybox_fs,
            mesh_pipeline,
            skybox_pipeline,
            cube_vertex_buffer,
            cube_index_buffer,
            sampler,
            environment_descriptor_set,
            materials_descriptor_set,
            materials_storage_buffer,
            environment_cubemap_query,
            meshes_with_materials_query,
            world,
        };

        (renderer, framebuffers)
    }

    pub fn resize(&mut self, images: Vec<Arc<Image>>) -> Vec<Arc<Framebuffer>> {
        let framebuffers;

        (framebuffers, (self.mesh_pipeline, self.skybox_pipeline)) = window_size_dependent_setup(
            &images,
            &self.render_pass,
            &self.memory_allocator,
            &self.mesh_vs,
            &self.mesh_fs,
            &self.skybox_vs,
            &self.skybox_fs,
            &self.sampler,
        );

        framebuffers
    }

    /// Returns true if the swapchain needs to be recreated
    pub fn draw(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        framebuffer: &Arc<Framebuffer>,
        camera: &Camera,
    ) {
        let timer = ProfileTimer::start("begin_render_pass");
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into()), Some(1f32.into())],

                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap();
        drop(timer);

        let aspect_ratio = framebuffer.extent()[0] as f32 / framebuffer.extent()[1] as f32;

        let timer = ProfileTimer::start("draw_meshes");
        self.draw_meshes(builder, &camera, aspect_ratio);
        drop(timer);

        let timer = ProfileTimer::start("next_subpass");
        builder
            .next_subpass(
                SubpassEndInfo::default(),
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap();
        drop(timer);

        let timer = ProfileTimer::start("draw_skybox");
        self.draw_skybox(builder, &camera, aspect_ratio);
        drop(timer);

        let timer = ProfileTimer::start("end_render_pass");
        builder.end_render_pass(Default::default()).unwrap();
        drop(timer);
    }

    fn draw_meshes(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        camera: &Camera,
        aspect_ratio: f32,
    ) {
        let frame_uniform_buffer = {
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
            self.mesh_pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::buffer(0, frame_uniform_buffer)],
            [],
        )
        .unwrap();

        let mut entities_data = vec![];
        let mut indirect_commands = vec![];

        let asset_database_read = self.asset_database.read().unwrap();
        self.meshes_with_materials_query
            .each(|(mesh_component, material, transform)| {
                let model_matrix = Mat4::from_scale_rotation_translation(
                    transform.scale,
                    transform.rotation,
                    transform.translation,
                );

                entities_data.push(mesh_shaders::vs::EntityData {
                    transform: model_matrix.to_cols_array_2d(),
                    material: material.material_id.0,
                    pad: [0; 3],
                });

                let mesh = asset_database_read
                    .get_mesh(mesh_component.mesh_id.clone())
                    .unwrap();
                indirect_commands.push(DrawIndexedIndirectCommand {
                    index_count: mesh.index_count,
                    instance_count: 1,
                    first_index: mesh.first_index,
                    vertex_offset: mesh.vertex_offset,
                    first_instance: 0,
                });
            });
        drop(asset_database_read);

        let indirect_buffer = self
            .indirect_buffer_allocator
            .allocate_slice(indirect_commands.len() as DeviceSize)
            .unwrap();
        indirect_buffer
            .write()
            .unwrap()
            .copy_from_slice(&indirect_commands);

        let entity_data_storage_buffer = self
            .storage_buffer_allocator
            .allocate_slice(entities_data.len() as DeviceSize)
            .unwrap();
        entity_data_storage_buffer
            .write()
            .unwrap()
            .copy_from_slice(&entities_data);

        let entity_data_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.mesh_pipeline.layout().set_layouts()[2].clone(),
            [WriteDescriptorSet::buffer(
                0,
                entity_data_storage_buffer.clone(),
            )],
            [],
        )
        .unwrap();

        builder
            .bind_pipeline_graphics(self.mesh_pipeline.clone())
            .unwrap();

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.mesh_pipeline.layout().clone(),
                0,
                (
                    frame_descriptor_set.clone(),
                    self.materials_descriptor_set.clone(),
                    entity_data_descriptor_set.clone(),
                    self.environment_descriptor_set.clone(),
                ),
            )
            .map_err(|err| {
                println!("{}", err);
                println!("{:#?}", *self.mesh_pipeline.layout().set_layouts()[3]);
                println!("{:#?}", self.environment_descriptor_set.layout());
            }).unwrap();

        let asset_database_read = self.asset_database.read().unwrap();
        builder
            .bind_vertex_buffers(0, asset_database_read.vertex_buffer().clone())
            .unwrap()
            .bind_index_buffer(asset_database_read.index_buffer().clone())
            .unwrap();
        drop(asset_database_read);

        unsafe { builder.draw_indexed_indirect(indirect_buffer) }.unwrap();
    }

    fn draw_skybox(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        camera: &Camera,
        aspect_ratio: f32,
    ) {
        builder
            .bind_pipeline_graphics(self.skybox_pipeline.clone())
            .unwrap();

        let mut environment_map = None;
        self.world
            .query::<&components::EnvironmentCubemap>()
            .term_at(0)
            .singleton()
            .build()
            .each(|env_cubemap| {
                environment_map = Some(env_cubemap.environment_map.clone());
            });
        let environment_map = environment_map.unwrap();

        let asset_database_read = self.asset_database.read().unwrap();
        let environment_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.skybox_pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::image_view(
                1,
                asset_database_read
                    .get_cubemap(environment_map)
                    .unwrap()
                    .cubemap
                    .clone(),
            )],
            [],
        )
        .unwrap();
        drop(asset_database_read);

        let frame_uniform_buffer = {
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
