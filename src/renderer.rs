use std::{
    sync::{Arc, RwLock},
    u32,
};

use flecs_ecs::prelude::*;
use glam::{Mat4, UVec3, Vec2, Vec3};
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        Buffer, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        DrawIndexedIndirectCommand, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
        SubpassBeginInfo, SubpassContents, SubpassEndInfo,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator,
        layout::{DescriptorBindingFlags, DescriptorType},
        DescriptorSet, WriteDescriptorSet,
    },
    device::{Device, DeviceOwned, Queue},
    format::Format,
    image::{
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, LOD_CLAMP_NONE},
        view::ImageView,
        Image, ImageCreateInfo, ImageType, ImageUsage,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo,
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
        ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::EntryPoint,
    DeviceSize,
};

use crate::{
    assets::{
        database::{
            AssetDatabase, AssetDatabaseChangeObserver, Cubemap, CubemapId, Material, MaterialId,
            Mesh, MeshId, Texture, TextureId,
        },
        loaders::mesh_loader::load_mesh_from_buffers_into_new_buffers,
        vertex::Vertex,
    },
    camera::Camera,
    ecs::components::{
        self, DirectionalLight, MaterialComponent, MeshComponent, PointLight, Transform,
    },
    profile::ProfileTimer,
};

const MAX_LIGHTS_PER_TILE: u32 = 64;
const TILE_SIZE: u32 = 32;
const Z_SLICES: u32 = 32;

/// The index of the descitptor set that doesnt change every frame
/// It has textures and materials buffer
pub const SLOW_CHANGING_DESCRIPTOR_SET: usize = 0;
pub const SLOW_CHANGING_DESCRIPTOR_SET_SAMPLER_BINDING: u32 = 0;
pub const SLOW_CHANGING_DESCRIPTOR_SET_MATERIALS_BUFFER_BINDING: u32 = 4;
pub const SLOW_CHANGING_DESCRIPTOR_SET_TEXTURES_BINDING: u32 = 5;

/// The index of the descriptor set that changes every frame
/// Has camera matrices and the transformations and materials indices for every entity
pub const FRAME_DESCRIPTOR_SET: usize = 1;
pub const FRAME_DESCRIPTOR_SET_CAMERA_MATRICES_BINDING: u32 = 0;
pub const FRAME_DESCRIPTOR_SET_ENTITY_DATA_BUFFER_BINDING: u32 = 1;
pub const FRAME_DESCRIPTOR_SET_DIRECTIONAL_LIGHTS_BUFFER_BINDING: u32 = 2;
pub const FRAME_DESCRIPTOR_SET_POINT_LIGHTS_BUFFER_BINDING: u32 = 3;
pub const FRAME_DESCRIPTOR_SET_VISIBLE_POINT_LIGHTS_BUFFER_BINDING: u32 = 4;
pub const FRAME_DESCRIPTOR_SET_POINT_LIGHTS_FROM_TILE_STORAGE_IMAGE_BINDING: u32 = 5;

pub struct Renderer {
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    uniform_buffer_allocator: SubbufferAllocator,
    host_writable_storage_buffer_allocator: SubbufferAllocator,
    in_device_storage_buffer_allocator: SubbufferAllocator,
    indirect_buffer_allocator: SubbufferAllocator,

    asset_database: Arc<RwLock<AssetDatabase>>,

    render_pass: Arc<RenderPass>,
    mesh_vs: EntryPoint,
    mesh_fs: EntryPoint,
    mesh_pipeline: Arc<GraphicsPipeline>,
    skybox_vs: EntryPoint,
    skybox_fs: EntryPoint,
    skybox_pipeline: Arc<GraphicsPipeline>,

    /// For tiled rendering
    light_culling_cs: EntryPoint,
    light_culling_pipeline: Arc<ComputePipeline>,

    cube_vertex_buffer: Subbuffer<[Vertex]>,
    cube_index_buffer: Subbuffer<[u32]>,

    asset_change_listener: Arc<RwLock<RendererAssetChangeListener>>,

    sampler: Arc<Sampler>,

    materials_storage_buffer: Subbuffer<[mesh_shaders::fs::Material]>,
    slow_changing_descriptor_set: Arc<DescriptorSet>,

    environment_cubemap_query: Query<&'static components::EnvironmentCubemap>,
    meshes_with_materials_query: Query<(
        &'static MeshComponent,
        &'static MaterialComponent,
        &'static Transform,
    )>,
    world: World,
}

#[derive(Debug)]
struct RendererAssetChangeListener {
    materials_changed: bool,
    textures_changed: bool,
}

fn window_size_dependent_setup(
    extent: Vec2,
    render_pass: &Arc<RenderPass>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    mesh_vs: &EntryPoint,
    mesh_fs: &EntryPoint,
    skybox_vs: &EntryPoint,
    skybox_fs: &EntryPoint,
    sampler: &Arc<Sampler>,
) -> (Arc<GraphicsPipeline>, Arc<GraphicsPipeline>) {
    let device = memory_allocator.device();

    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: extent.into(),
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

        // SLOW_CHANGING_DESCRIPTOR_SET
        layout_create_info.set_layouts[SLOW_CHANGING_DESCRIPTOR_SET]
            .bindings
            .get_mut(&SLOW_CHANGING_DESCRIPTOR_SET_SAMPLER_BINDING)
            .unwrap()
            .immutable_samplers = vec![sampler.clone()];
        layout_create_info.set_layouts[SLOW_CHANGING_DESCRIPTOR_SET]
            .bindings
            .get_mut(&SLOW_CHANGING_DESCRIPTOR_SET_TEXTURES_BINDING)
            .unwrap()
            .descriptor_count = properties.max_descriptor_set_samplers - 3;
        layout_create_info.set_layouts[SLOW_CHANGING_DESCRIPTOR_SET]
            .bindings
            .get_mut(&SLOW_CHANGING_DESCRIPTOR_SET_TEXTURES_BINDING)
            .unwrap()
            .binding_flags = DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
            | DescriptorBindingFlags::PARTIALLY_BOUND;

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

    (mesh_pipeline, skybox_pipeline)
}

impl AssetDatabaseChangeObserver for RendererAssetChangeListener {
    fn on_mesh_add(&mut self, _mesh_id: MeshId, _mesh: &Mesh) {
        // For now nothing needs to be done here
    }

    fn on_texture_add(&mut self, _texture_id: TextureId, _texture: &Texture) {
        // TODO: in the future create an allocator of indices that allocates an index for each id
        // so we can use random ids for assets and not use the id as index (ids are secuential for
        // now but that could change and texture indices need to be mostly contiguous)
        //
        // TODO: Maybe add the added meshes to a queue and then add them in the begining of the
        // next frame (in that case we wont need the &Texture reference). This would be to not
        // recreate the environment_descriptor_set multiple times per frame and to not update the
        // descriptor set after binding
        self.textures_changed = true;
    }

    fn on_cubemap_add(&mut self, _cubemap_id: CubemapId, _cubemap: &Cubemap) {
        // For now nothing needs to be done here
    }

    fn on_material_add(&mut self, _material_id: MaterialId, _material: &Material) {
        self.materials_changed = true;
    }
}

impl Renderer {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        asset_database: Arc<RwLock<AssetDatabase>>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        world: World,
        format: Format,
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

        let host_writable_storage_buffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::STORAGE_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let in_device_storage_buffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
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

        let light_culling_cs = light_culling::cs::load(device.clone())
            .unwrap()
            .specialize(
                [
                    (0, MAX_LIGHTS_PER_TILE.into()),
                    (1, TILE_SIZE.into()),
                    (2, Z_SLICES.into()),
                ]
                .into_iter()
                .collect(),
            )
            .unwrap()
            .entry_point("main")
            .unwrap();
        let stage = PipelineShaderStageCreateInfo::new(light_culling_cs.clone());
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        let light_culling_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .unwrap();

        let render_pass = vulkano::ordered_passes_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: format,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
                depth_stencil: {
                    format: Format::D32_SFLOAT,
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
            .specialize(
                [(0, TILE_SIZE.into()), (1, Z_SLICES.into())]
                    .into_iter()
                    .collect(),
            )
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

        let (mesh_pipeline, skybox_pipeline) = window_size_dependent_setup(
            Vec2::new(1280.0, 720.0),
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

        assert!(size_of::<mesh_shaders::fs::Material>() % 16 == 0);

        let materials_storage_buffer = Buffer::new_slice(
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
            1,
        )
        .unwrap();

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
        let environment_descriptor_set = DescriptorSet::new_variable(
            descriptor_set_allocator.clone(),
            mesh_pipeline.layout().set_layouts()[SLOW_CHANGING_DESCRIPTOR_SET].clone(),
            1, // If this is zero for some reason it cant be deallocated
            [
                WriteDescriptorSet::image_view(
                    1,
                    asset_database_read
                        .get_cubemap(irradiance_map.clone())
                        .unwrap()
                        .cubemap
                        .clone(),
                ),
                WriteDescriptorSet::image_view(
                    2,
                    asset_database_read
                        .get_cubemap(prefiltered_environment_map.clone())
                        .unwrap()
                        .cubemap
                        .clone(),
                ),
                WriteDescriptorSet::image_view(
                    3,
                    asset_database_read
                        .get_cubemap(environment_brdf_lut.clone())
                        .unwrap()
                        .cubemap
                        .clone(),
                ),
                WriteDescriptorSet::buffer(
                    SLOW_CHANGING_DESCRIPTOR_SET_MATERIALS_BUFFER_BINDING,
                    materials_storage_buffer.clone(),
                ),
                // WriteDescriptorSet::image_view_array(
                //     4,
                //     0,
                //     asset_database_read
                //         .textures()
                //         .iter()
                //         .map(|texture| texture.texture.clone()),
                // ),
            ],
            [],
        )
        .unwrap();
        drop(asset_database_read);

        let asset_change_listener = Arc::new(RwLock::new(RendererAssetChangeListener {
            textures_changed: false,
            materials_changed: false,
        }));

        asset_database
            .write()
            .unwrap()
            .add_asset_database_change_observer(asset_change_listener.clone());

        let renderer = Self {
            memory_allocator,
            descriptor_set_allocator,
            uniform_buffer_allocator,
            host_writable_storage_buffer_allocator,
            in_device_storage_buffer_allocator,
            indirect_buffer_allocator,
            asset_database,
            render_pass,
            mesh_vs,
            mesh_fs,
            skybox_vs,
            skybox_fs,
            mesh_pipeline,
            skybox_pipeline,
            light_culling_cs,
            light_culling_pipeline,
            cube_vertex_buffer,
            cube_index_buffer,
            sampler,
            materials_storage_buffer,
            slow_changing_descriptor_set: environment_descriptor_set,
            asset_change_listener,
            environment_cubemap_query,
            meshes_with_materials_query,
            world,
        };

        renderer
    }

    fn add_materials_and_textures_to_descriptor_set(&mut self) {
        let asset_change_listener_read = self.asset_change_listener.read().unwrap();
        let textures_changed = asset_change_listener_read.textures_changed;
        let materials_changed = asset_change_listener_read.materials_changed;
        drop(asset_change_listener_read);

        if !textures_changed && !materials_changed {
            return;
        }

        if materials_changed {
            let asset_database_read = self.asset_database.read().unwrap();

            let materials: Vec<_> = asset_database_read
                .materials()
                .iter()
                .map(|material| {
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
                })
                .collect();

            self.materials_storage_buffer = self
                .host_writable_storage_buffer_allocator
                .allocate_slice(materials.len() as DeviceSize)
                .unwrap();
            self.materials_storage_buffer
                .write()
                .unwrap()
                .copy_from_slice(&materials);
        }

        let mut irradiance_map = None;
        let mut prefiltered_environment_map = None;
        let mut environment_brdf_lut = None;

        self.environment_cubemap_query.each(|env_cubemap| {
            irradiance_map = Some(env_cubemap.irradiance_map.clone());
            prefiltered_environment_map = Some(env_cubemap.prefiltered_environment_map.clone());
            environment_brdf_lut = Some(env_cubemap.environment_brdf_lut.clone());
        });
        let irradiance_map = irradiance_map.unwrap();
        let prefiltered_environment_map = prefiltered_environment_map.unwrap();
        let environment_brdf_lut = environment_brdf_lut.unwrap();

        let asset_database_read = self.asset_database.read().unwrap();
        self.slow_changing_descriptor_set = DescriptorSet::new_variable(
            self.descriptor_set_allocator.clone(),
            self.mesh_pipeline.layout().set_layouts()[SLOW_CHANGING_DESCRIPTOR_SET].clone(),
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
                WriteDescriptorSet::buffer(
                    SLOW_CHANGING_DESCRIPTOR_SET_MATERIALS_BUFFER_BINDING,
                    self.materials_storage_buffer.clone(),
                ),
                WriteDescriptorSet::image_view_array(
                    SLOW_CHANGING_DESCRIPTOR_SET_TEXTURES_BINDING,
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
        drop(asset_database_read);

        let mut asset_change_listener_write = self.asset_change_listener.write().unwrap();
        asset_change_listener_write.materials_changed = false;
        asset_change_listener_write.textures_changed = false;
    }

    fn create_point_lights_storage_buffer(
        &mut self
    ) -> Option<Subbuffer<[mesh_shaders::fs::PointLight]>> {
        let mut point_lights = vec![];
        self.world
            .each::<(&PointLight, &Transform)>(|(point_light, transform)| {
                point_lights.push(mesh_shaders::fs::PointLight {
                    position: transform.translation.to_array().into(),
                    radius: point_light.radius,
                    color: point_light.color.into(),
                    pad: [0; 1],
                });
            });

        if point_lights.is_empty() {
            None
        } else {
            let point_lights_storage_buffer = self.host_writable_storage_buffer_allocator
                .allocate_slice(point_lights.len() as DeviceSize)
                .unwrap();

            point_lights_storage_buffer
                .write()
                .unwrap()
                .copy_from_slice(&point_lights);

            Some(point_lights_storage_buffer)
        }
    }

    fn create_directional_lights_storage_buffer(
        &mut self
    ) -> Option<Subbuffer<[mesh_shaders::fs::DirectionalLight]>> {
        let mut directional_lights = vec![];
        self.world
            .each::<(&DirectionalLight, &Transform)>(|(directional_light, transform)| {
                directional_lights.push(mesh_shaders::fs::DirectionalLight {
                    direction: (transform.rotation * Vec3::NEG_Z).to_array().into(),
                    color: directional_light.color.into(),
                    pad: [0; 1],
                });
            });

        if directional_lights.is_empty() {
            None
        } else {
            let directional_lights_storage_buffer = self.host_writable_storage_buffer_allocator
                .allocate_slice(directional_lights.len() as DeviceSize)
                .unwrap();

            directional_lights_storage_buffer
                .write()
                .unwrap()
                .copy_from_slice(&directional_lights);

            Some(directional_lights_storage_buffer)
        }
    }

    fn create_framebuffers(&self, image_views: &[Arc<ImageView>]) -> Vec<Arc<Framebuffer>> {
        let extent = image_views[0].image().extent();

        let depth_buffer = ImageView::new_default(
            Image::new(
                self.memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::D32_SFLOAT,
                    extent,
                    usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        let framebuffers = image_views
            .iter()
            .map(|image_view| {
                Framebuffer::new(
                    self.render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![image_view.clone(), depth_buffer.clone()],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        framebuffers
    }

    pub fn resize_and_create_framebuffers(
        &mut self,
        image_views: &[Arc<ImageView>],
    ) -> Vec<Arc<Framebuffer>> {
        let extent = Vec2::new(
            image_views[0].image().extent()[0] as f32,
            image_views[0].image().extent()[1] as f32,
        );
        (self.mesh_pipeline, self.skybox_pipeline) = window_size_dependent_setup(
            extent,
            &self.render_pass,
            &self.memory_allocator,
            &self.mesh_vs,
            &self.mesh_fs,
            &self.skybox_vs,
            &self.skybox_fs,
            &self.sampler,
        );

        self.create_framebuffers(&image_views)
    }

    /// Returns true if the swapchain needs to be recreated
    pub fn draw(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        framebuffer: &Arc<Framebuffer>,
        camera: &Camera,
    ) {
        self.add_materials_and_textures_to_descriptor_set();
        let point_lights_buffer = self.create_point_lights_storage_buffer();
        let directional_lights_buffer = self.create_directional_lights_storage_buffer();

        let aspect_ratio = framebuffer.extent()[0] as f32 / framebuffer.extent()[1] as f32;

        let timer = ProfileTimer::start("cull_lights");
        let (visible_light_indices_storage_buffer, lights_from_tile_storage_image_view) =
            self.cull_lights(
                builder,
                &camera,
                aspect_ratio,
                &point_lights_buffer,
            );
        drop(timer);

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

        let timer = ProfileTimer::start("draw_meshes");
        self.draw_meshes(
            builder,
            &camera,
            aspect_ratio,
            &point_lights_buffer,
            &directional_lights_buffer,
            visible_light_indices_storage_buffer,
            lights_from_tile_storage_image_view,
        );
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

    fn cull_lights(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        camera: &Camera,
        aspect_ratio: f32,
        point_lights_storage_buffer: &Option<Subbuffer<[mesh_shaders::fs::PointLight]>>,
    ) -> (Subbuffer<[u32]>, Arc<ImageView>) {
        let next_ligth_index_global_storage_buffer = self
            .in_device_storage_buffer_allocator
            .allocate_slice::<light_culling::cs::NextLigthIndexGlobal>(4) // allocating 1 freezes
            .unwrap();

        // Initialize with value 0
        assert_eq!(
            size_of::<light_culling::cs::NextLigthIndexGlobal>(),
            size_of::<u32>()
        );
        builder
            .fill_buffer(
                next_ligth_index_global_storage_buffer
                    .clone()
                    .reinterpret::<[u32]>(),
                0,
            )
            .unwrap();

        let width = self
            .mesh_pipeline
            .viewport_state()
            .unwrap()
            .viewports
            .get(0)
            .unwrap()
            .extent[0] as u32;
        let height = self
            .mesh_pipeline
            .viewport_state()
            .unwrap()
            .viewports
            .get(0)
            .unwrap()
            .extent[1] as u32;

        let num_tiles = UVec3::new(
            width.div_ceil(TILE_SIZE),
            height.div_ceil(TILE_SIZE),
            Z_SLICES,
        );

        let visible_light_indices_storage_buffer = self
            .in_device_storage_buffer_allocator
            .allocate_slice::<u32>(
                (num_tiles.x * num_tiles.y * num_tiles.z * MAX_LIGHTS_PER_TILE) as DeviceSize,
            )
            .unwrap();

        let lights_from_tile_storage_image = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim3d,
                format: Format::R32G32_UINT,
                extent: num_tiles.into(),
                usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();

        let lights_from_tile_storage_image_view =
            ImageView::new_default(lights_from_tile_storage_image).unwrap();

        let frame_uniform_buffer = {
            let view = Mat4::look_to_rh(
                camera.position,
                camera.rotation * Vec3::NEG_Z,
                camera.rotation * Vec3::Y,
            );
            let proj = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0))
                * Mat4::perspective_rh_gl(camera.fov, aspect_ratio, 0.01, 100.0);

            let num_lights = match point_lights_storage_buffer {
                Some(point_lights_storage_buffer) => point_lights_storage_buffer.len() as _,
                None => 0,
            };

            let uniform_data = light_culling::cs::FrameUniforms {
                view: view.to_cols_array_2d(),
                view_proj: (proj * view).to_cols_array_2d(),
                num_lights,
                width: width as f32,
                height: height as f32,
                near: 0.01,
                far: 100.0,
            };

            let buffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
            *buffer.write().unwrap() = uniform_data;

            buffer
        };

        let set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.light_culling_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, frame_uniform_buffer.clone()),
                WriteDescriptorSet::buffer(
                    1,
                    point_lights_storage_buffer
                        .as_ref()
                        .unwrap_or(
                            &self
                                .in_device_storage_buffer_allocator
                                .allocate_slice(1)
                                .unwrap(),
                        )
                        .clone(),
                ),
                WriteDescriptorSet::buffer(2, next_ligth_index_global_storage_buffer.clone()),
                WriteDescriptorSet::buffer(3, visible_light_indices_storage_buffer.clone()),
                WriteDescriptorSet::image_view(4, lights_from_tile_storage_image_view.clone()),
            ],
            [],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.light_culling_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.light_culling_pipeline.layout().clone(),
                0,
                set,
            )
            .unwrap();

        unsafe { builder.dispatch(num_tiles.to_array()) }.unwrap();

        (
            visible_light_indices_storage_buffer,
            lights_from_tile_storage_image_view,
        )
    }

    fn draw_meshes(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        camera: &Camera,
        aspect_ratio: f32,
        point_lights_storage_buffer: &Option<Subbuffer<[mesh_shaders::fs::PointLight]>>,
        directional_lights_storage_buffer: &Option<Subbuffer<[mesh_shaders::fs::DirectionalLight]>>,
        visible_light_indices_storage_buffer: Subbuffer<[u32]>,
        lights_from_tile_storage_image_view: Arc<ImageView>,
    ) {
        let view = Mat4::look_to_rh(
            camera.position,
            camera.rotation * Vec3::NEG_Z,
            camera.rotation * Vec3::Y,
        );

        let frame_uniform_buffer = {
            let width = self
                .mesh_pipeline
                .viewport_state()
                .unwrap()
                .viewports
                .get(0)
                .unwrap()
                .extent[0];
            let height = self
                .mesh_pipeline
                .viewport_state()
                .unwrap()
                .viewports
                .get(0)
                .unwrap()
                .extent[1];
            let proj = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0))
                * Mat4::perspective_rh_gl(camera.fov, aspect_ratio, 0.01, 100.0);

            let uniform_data = mesh_shaders::vs::FrameUniforms {
                view: view.to_cols_array_2d(),
                proj: proj.to_cols_array_2d(),
                view_position: camera.position.into(),
                near: 0.01,
                far: 100.0,
                width,
                height,
                directional_light_count: directional_lights_storage_buffer
                    .as_ref()
                    .map(|buffer| buffer.len() as u32)
                    .unwrap_or(0),
            };

            let buffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
            *buffer.write().unwrap() = uniform_data;

            buffer
        };

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

        if indirect_commands.is_empty() {
            return;
        }

        let indirect_buffer = self
            .indirect_buffer_allocator
            .allocate_slice(indirect_commands.len() as DeviceSize)
            .unwrap();
        indirect_buffer
            .write()
            .unwrap()
            .copy_from_slice(&indirect_commands);

        let entity_data_storage_buffer = self
            .host_writable_storage_buffer_allocator
            .allocate_slice(entities_data.len() as DeviceSize)
            .unwrap();
        entity_data_storage_buffer
            .write()
            .unwrap()
            .copy_from_slice(&entities_data);

        let frame_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.mesh_pipeline.layout().set_layouts()[FRAME_DESCRIPTOR_SET].clone(),
            [
                WriteDescriptorSet::buffer(
                    FRAME_DESCRIPTOR_SET_CAMERA_MATRICES_BINDING,
                    frame_uniform_buffer,
                ),
                WriteDescriptorSet::buffer(
                    FRAME_DESCRIPTOR_SET_ENTITY_DATA_BUFFER_BINDING,
                    entity_data_storage_buffer.clone(),
                ),
                WriteDescriptorSet::buffer(
                    FRAME_DESCRIPTOR_SET_DIRECTIONAL_LIGHTS_BUFFER_BINDING,
                    directional_lights_storage_buffer
                        .as_ref()
                        .unwrap_or(
                            &self
                                .in_device_storage_buffer_allocator
                                .allocate_slice(1)
                                .unwrap(),
                        )
                        .clone(),
                ),
                WriteDescriptorSet::buffer(
                    FRAME_DESCRIPTOR_SET_POINT_LIGHTS_BUFFER_BINDING,
                    point_lights_storage_buffer
                        .as_ref()
                        .unwrap_or(
                            &self
                                .in_device_storage_buffer_allocator
                                .allocate_slice(1)
                                .unwrap(),
                        )
                        .clone(),
                ),
                WriteDescriptorSet::buffer(
                    FRAME_DESCRIPTOR_SET_VISIBLE_POINT_LIGHTS_BUFFER_BINDING,
                    visible_light_indices_storage_buffer.clone(),
                ),
                WriteDescriptorSet::image_view(
                    FRAME_DESCRIPTOR_SET_POINT_LIGHTS_FROM_TILE_STORAGE_IMAGE_BINDING,
                    lights_from_tile_storage_image_view.clone(),
                ),
            ],
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
                    self.slow_changing_descriptor_set.clone(),
                    frame_descriptor_set.clone(),
                ),
            )
            .unwrap();

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

mod light_culling {
    pub mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "assets/shaders/light_culling.cs.glsl",
        }
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
