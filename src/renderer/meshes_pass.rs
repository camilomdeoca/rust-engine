use std::sync::{Arc, RwLock};

use flecs_ecs::prelude::*;
use glam::{Mat4, UVec2, Vec2, Vec3};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, DrawIndexedIndirectCommand, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassEndInfo,
    },
    descriptor_set::{
        layout::{DescriptorBindingFlags, DescriptorType},
        DescriptorSet, WriteDescriptorSet,
    },
    device::Device,
    format::Format,
    image::{
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, LOD_CLAMP_NONE},
        view::{ImageView, ImageViewCreateInfo},
        Image, ImageCreateInfo, ImageSubresourceRange, ImageType, ImageUsage,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
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
    DeviceSize, Validated, VulkanError,
};

use crate::{
    assets::{
        database::AssetDatabase, loaders::mesh_loader::load_mesh_from_buffers_into_new_buffers,
        vertex::Vertex,
    },
    camera::Camera,
    ecs::components::{self, MaterialComponent, MeshComponent, Transform},
    renderer::RendererAssetChangeListener,
    settings::{Settings, SHADOW_MAP_CASCADE_COUNT},
};

use super::{RendererContext, RendererError};

/// The index of the descitptor set that doesnt change every frame
/// It has textures and materials buffer
pub const SLOW_CHANGING_DESCRIPTOR_SET: usize = 0;
pub const SLOW_CHANGING_DESCRIPTOR_SET_SAMPLER_BINDING: u32 = 0;
pub const SLOW_CHANGING_DESCRIPTOR_SET_SHADOW_MAP_SAMPLER_BINDING: u32 = 1;
pub const SLOW_CHANGING_DESCRIPTOR_SET_MATERIALS_BUFFER_BINDING: u32 = 5;
pub const SLOW_CHANGING_DESCRIPTOR_SET_TEXTURES_BINDING: u32 = 6;

/// The index of the descriptor set that changes every frame
/// Has camera matrices and the transformations and materials indices for every entity
pub const FRAME_DESCRIPTOR_SET: usize = 1;
pub const FRAME_DESCRIPTOR_SET_CAMERA_MATRICES_BINDING: u32 = 0;
pub const FRAME_DESCRIPTOR_SET_ENTITY_DATA_BUFFER_BINDING: u32 = 1;
pub const FRAME_DESCRIPTOR_SET_DIRECTIONAL_LIGHTS_BUFFER_BINDING: u32 = 2;
pub const FRAME_DESCRIPTOR_SET_POINT_LIGHTS_BUFFER_BINDING: u32 = 3;
pub const FRAME_DESCRIPTOR_SET_VISIBLE_POINT_LIGHTS_BUFFER_BINDING: u32 = 4;
pub const FRAME_DESCRIPTOR_SET_POINT_LIGHTS_FROM_TILE_STORAGE_IMAGE_BINDING: u32 = 5;
pub const FRAME_DESCRIPTOR_SET_SHADOW_MAP_BINDING: u32 = 6;

pub struct MeshesPass {
    render_pass: Arc<RenderPass>,

    sampler: Arc<Sampler>,
    shadow_map_sampler: Arc<Sampler>,

    mesh_vs: EntryPoint,
    mesh_fs: EntryPoint,
    mesh_pipeline: Arc<GraphicsPipeline>,

    skybox_vs: EntryPoint,
    skybox_fs: EntryPoint,
    skybox_pipeline: Arc<GraphicsPipeline>,

    materials_storage_buffer: Subbuffer<[mesh_shaders::fs::Material]>,
    slow_changing_descriptor_set: Arc<DescriptorSet>,

    environment_cubemap_query: Query<&'static components::EnvironmentCubemap>,
    meshes_with_materials_query: Query<(
        &'static MeshComponent,
        &'static MaterialComponent,
        &'static Transform,
    )>,

    cube_vertex_buffer: Subbuffer<[Vertex]>,
    cube_index_buffer: Subbuffer<[u32]>,

    asset_change_listener: Arc<RwLock<RendererAssetChangeListener>>,
}

pub fn create_main_render_pass(
    device: &Arc<Device>,
) -> Result<Arc<RenderPass>, Validated<VulkanError>> {
    vulkano::ordered_passes_renderpass!(
        device.clone(),
        attachments: {
            g_buffer: {
                format: Format::R32G32_UINT,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
            depth_stencil: {
                format: Format::D32_SFLOAT,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        passes: [
            // Mesh rendering
            {
                color: [g_buffer],
                depth_stencil: {depth_stencil},
                input: []
            },
            // Skybox (after so it doesnt draw behind the meshes)
            {
                color: [g_buffer],
                depth_stencil: {depth_stencil},
                input: []
            }
        ]
    )
}

pub fn create_mesh_pipeline(
    extent: Vec2,
    mesh_pass: Subpass,
    device: &Arc<Device>,
    mesh_vs: &EntryPoint,
    mesh_fs: &EntryPoint,
    sampler: &Arc<Sampler>,
    shadow_map_sampler: &Arc<Sampler>,
) -> Result<Arc<GraphicsPipeline>, Validated<VulkanError>> {
    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: extent.into(),
        depth_range: 0.0..=1.0,
    };

    let properties = device.physical_device().properties();

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
        .get_mut(&SLOW_CHANGING_DESCRIPTOR_SET_SHADOW_MAP_SAMPLER_BINDING)
        .unwrap()
        .immutable_samplers = vec![shadow_map_sampler.clone()];

    layout_create_info.set_layouts[SLOW_CHANGING_DESCRIPTOR_SET]
        .bindings
        .get_mut(&SLOW_CHANGING_DESCRIPTOR_SET_TEXTURES_BINDING)
        .unwrap()
        .descriptor_count = properties.max_descriptor_set_samplers - 5;
    layout_create_info.set_layouts[SLOW_CHANGING_DESCRIPTOR_SET]
        .bindings
        .get_mut(&SLOW_CHANGING_DESCRIPTOR_SET_TEXTURES_BINDING)
        .unwrap()
        .binding_flags =
        DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT | DescriptorBindingFlags::PARTIALLY_BOUND;

    let layout = PipelineLayout::new(
        device.clone(),
        layout_create_info
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )?;

    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: mesh_pipeline_stages.into_iter().collect(),
            vertex_input_state: Some(Vertex::per_vertex().definition(mesh_vs)?),
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
}

pub fn create_skybox_pipeline(
    extent: Vec2,
    skybox_pass: Subpass,
    device: &Arc<Device>,
    skybox_vs: &EntryPoint,
    skybox_fs: &EntryPoint,
    sampler: &Arc<Sampler>,
) -> Result<Arc<GraphicsPipeline>, Validated<VulkanError>> {
    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: extent.into(),
        depth_range: 0.0..=1.0,
    };

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
    )?;

    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(Vertex::per_vertex().definition(skybox_vs)?),
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
}

fn create_cube_vertex_and_index_buffer(
    context: &RendererContext,
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) -> (Subbuffer<[Vertex]>, Subbuffer<[u32]>) {
    load_mesh_from_buffers_into_new_buffers(
        context.memory_allocator.clone(),
        builder,
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
    .unwrap()
}

impl MeshesPass {
    pub fn new(
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        context: &RendererContext,
        extent: Vec2,
        asset_database: &Arc<RwLock<AssetDatabase>>,
        world: &World,
    ) -> Result<Self, RendererError> {
        let render_pass = create_main_render_pass(&context.device)?;

        let sampler = Sampler::new(
            context.device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                lod: 0.0..=LOD_CLAMP_NONE,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )?;

        let shadow_map_sampler = Sampler::new(
            context.device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                lod: 0.0..=LOD_CLAMP_NONE,
                address_mode: [SamplerAddressMode::Repeat; 3],
                compare: Some(CompareOp::Less),
                ..Default::default()
            },
        )?;

        let mesh_vs = mesh_shaders::vs::load(context.device.clone())?
            .entry_point("main")
            .unwrap();
        let mesh_fs = mesh_shaders::fs::load(context.device.clone())?
            .entry_point("main")
            .unwrap();

        let mesh_pipeline = create_mesh_pipeline(
            extent,
            Subpass::from(render_pass.clone(), 0).unwrap(),
            &context.device,
            &mesh_vs,
            &mesh_fs,
            &sampler,
            &shadow_map_sampler,
        )?;

        let skybox_vs = skybox_shaders::vs::load(context.device.clone())?
            .entry_point("main")
            .unwrap();
        let skybox_fs = skybox_shaders::fs::load(context.device.clone())?
            .entry_point("main")
            .unwrap();

        let skybox_pipeline = create_skybox_pipeline(
            extent,
            Subpass::from(render_pass.clone(), 1).unwrap(),
            &context.device,
            &skybox_vs,
            &skybox_fs,
            &sampler,
        )?;

        let environment_cubemap_query = world
            .query::<&components::EnvironmentCubemap>()
            .term_at(0)
            .singleton()
            .build();

        let meshes_with_materials_query =
            world.new_query::<(&MeshComponent, &MaterialComponent, &Transform)>();

        assert!(size_of::<mesh_shaders::fs::Material>() % 16 == 0);

        let materials_storage_buffer = Buffer::new_slice(
            context.memory_allocator.clone(),
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
        let slow_changing_descriptor_set = DescriptorSet::new_variable(
            context.descriptor_set_allocator.clone(),
            mesh_pipeline.layout().set_layouts()[SLOW_CHANGING_DESCRIPTOR_SET].clone(),
            1, // If this is zero for some reason it cant be deallocated
            [
                WriteDescriptorSet::image_view(
                    2,
                    asset_database_read
                        .get_cubemap(irradiance_map.clone())
                        .unwrap()
                        .cubemap
                        .clone(),
                ),
                WriteDescriptorSet::image_view(
                    3,
                    asset_database_read
                        .get_cubemap(prefiltered_environment_map.clone())
                        .unwrap()
                        .cubemap
                        .clone(),
                ),
                WriteDescriptorSet::image_view(
                    4,
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
        )?;
        drop(asset_database_read);

        let asset_change_listener = Arc::new(RwLock::new(RendererAssetChangeListener {
            textures_changed: false,
            materials_changed: false,
        }));

        asset_database
            .write()
            .unwrap()
            .add_asset_database_change_observer(asset_change_listener.clone());

        let (cube_vertex_buffer, cube_index_buffer) =
            create_cube_vertex_and_index_buffer(context, builder);

        Ok(Self {
            render_pass,
            sampler,
            shadow_map_sampler,
            mesh_vs,
            mesh_fs,
            mesh_pipeline,
            skybox_vs,
            skybox_fs,
            skybox_pipeline,
            materials_storage_buffer,
            slow_changing_descriptor_set,
            environment_cubemap_query,
            meshes_with_materials_query,
            cube_vertex_buffer,
            cube_index_buffer,
            asset_change_listener,
        })
    }

    pub fn create_g_buffer_framebuffer(
        &self,
        context: &RendererContext,
        extent: UVec2,
    ) -> Result<Arc<Framebuffer>, RendererError> {
        // We need to build mip chain for rendering SSAO at lower resolution and to then upscale it
        // with Joint Bilateral upsampling. We will also use it to sample smaller levels first:
        // We will store the min of 4 pixels in each next mip level, and if when comparing the depth it
        // is outside the radius then we dont sample bigger level as all will be bigger (they are
        // behind)
        let mip_levels = extent.max_element().ilog2();
        let depth_buffer_image = Image::new(
            context.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::D32_SFLOAT,
                extent: [extent.x, extent.y, 1],
                mip_levels,
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();

        let depth_buffer = ImageView::new(
            depth_buffer_image.clone(),
            ImageViewCreateInfo {
                subresource_range: ImageSubresourceRange {
                    mip_levels: 0..1,
                    ..depth_buffer_image.subresource_range()
                },
                ..ImageViewCreateInfo::from_image(&depth_buffer_image)
            },
        )?;

        let g_buffer = ImageView::new_default(
            Image::new(
                context.memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R32G32_UINT,
                    extent: [extent.x, extent.y, 1],
                    usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )?;

        Ok(Framebuffer::new(
            self.render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![g_buffer.clone(), depth_buffer.clone()],
                ..Default::default()
            },
        )?)
    }

    pub fn resize_and_create_g_buffer(
        &mut self,
        context: &RendererContext,
        extent: UVec2,
    ) -> Result<Arc<Framebuffer>, RendererError> {
        let g_buffer = self.create_g_buffer_framebuffer(context, extent)?;

        self.mesh_pipeline = create_mesh_pipeline(
            extent.as_vec2(),
            Subpass::from(self.render_pass.clone(), 0).unwrap(),
            &context.device,
            &self.mesh_vs,
            &self.mesh_fs,
            &self.sampler,
            &self.shadow_map_sampler,
        )?;

        self.skybox_pipeline = create_skybox_pipeline(
            extent.as_vec2(),
            Subpass::from(self.render_pass.clone(), 1).unwrap(),
            &context.device,
            &self.skybox_vs,
            &self.skybox_fs,
            &self.sampler,
        )?;

        Ok(g_buffer)
    }

    fn add_materials_and_textures_to_descriptor_set(
        &mut self,
        context: &RendererContext,
        asset_database: &Arc<RwLock<AssetDatabase>>,
    ) -> Result<(), RendererError> {
        let asset_change_listener_read = self.asset_change_listener.read().unwrap();
        let textures_changed = asset_change_listener_read.textures_changed;
        let materials_changed = asset_change_listener_read.materials_changed;
        drop(asset_change_listener_read);

        if !textures_changed && !materials_changed {
            return Ok(());
        }

        if materials_changed {
            let asset_database_read = asset_database.read().unwrap();

            let materials: Vec<_> = asset_database_read
                .materials()
                .iter()
                .map(|material| {
                    let diffuse = material.diffuse.clone().map(|id| id.0);
                    let metallic_roughness = material.metallic_roughness.clone().map(|id| id.0);
                    let ambient_occlusion = material.ambient_occlusion.clone().map(|id| id.0);
                    let emissive = material.emissive.clone().map(|id| id.0);
                    let normal = material.normal.clone().map(|id| id.0);
                    mesh_shaders::fs::Material {
                        base_color_factor: material.color_factor.into(),
                        emissive_factor: material.emissive_factor.into(),
                        metallic_factor: material.metallic_factor.into(),
                        roughness_factor: material.roughness_factor.into(),
                        base_color_texture_id: diffuse.unwrap_or(u32::MAX),
                        metallic_roughness_texture_id: metallic_roughness.unwrap_or(u32::MAX),
                        ambient_occlusion_texture_id: ambient_occlusion.unwrap_or(u32::MAX).into(),
                        emissive_texture_id: emissive.unwrap_or(u32::MAX),
                        normal_texture_id: normal.unwrap_or(u32::MAX),
                        pad: [0; 3],
                    }
                })
                .collect();

            self.materials_storage_buffer = context
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

        let asset_database_read = asset_database.read().unwrap();
        self.slow_changing_descriptor_set = DescriptorSet::new_variable(
            context.descriptor_set_allocator.clone(),
            self.mesh_pipeline.layout().set_layouts()[SLOW_CHANGING_DESCRIPTOR_SET].clone(),
            asset_database_read.textures().len() as u32,
            [
                WriteDescriptorSet::image_view(
                    2,
                    asset_database_read
                        .get_cubemap(irradiance_map)
                        .unwrap()
                        .cubemap
                        .clone(),
                ),
                WriteDescriptorSet::image_view(
                    3,
                    asset_database_read
                        .get_cubemap(prefiltered_environment_map)
                        .unwrap()
                        .cubemap
                        .clone(),
                ),
                WriteDescriptorSet::image_view(
                    4,
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
        )?;
        drop(asset_database_read);

        let mut asset_change_listener_write = self.asset_change_listener.write().unwrap();
        asset_change_listener_write.materials_changed = false;
        asset_change_listener_write.textures_changed = false;
        Ok(())
    }

    // TODO: add frustum culling here
    /// Returns None if there isnt entities with meshes and materials
    fn get_entities_data_and_indirect_draw_commands(
        &self,
        context: &RendererContext,
        asset_database: &Arc<RwLock<AssetDatabase>>,
    ) -> Option<(
        Subbuffer<[mesh_shaders::vs::EntityData]>,
        Subbuffer<[DrawIndexedIndirectCommand]>,
    )> {
        let mut entities_data = vec![];
        let mut indirect_commands = vec![];

        let asset_database_read = asset_database.read().unwrap();
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
            return None;
        }

        let indirect_buffer = context
            .indirect_buffer_allocator
            .allocate_slice(indirect_commands.len() as DeviceSize)
            .unwrap();
        indirect_buffer
            .write()
            .unwrap()
            .copy_from_slice(&indirect_commands);

        let entity_data_storage_buffer = context
            .host_writable_storage_buffer_allocator
            .allocate_slice(entities_data.len() as DeviceSize)
            .unwrap();
        entity_data_storage_buffer
            .write()
            .unwrap()
            .copy_from_slice(&entities_data);
        Some((entity_data_storage_buffer, indirect_buffer))
    }

    pub fn run(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        context: &RendererContext,
        camera: &Camera,
        aspect_ratio: f32,
        point_lights_storage_buffer: &Option<Subbuffer<[mesh_shaders::fs::PointLight]>>,
        directional_lights_storage_buffer: &Option<Subbuffer<[mesh_shaders::fs::DirectionalLight]>>,
        visible_light_indices_storage_buffer: &Subbuffer<[u32]>,
        lights_from_tile_storage_image_view: &Arc<ImageView>,
        shadow_map: &Arc<ImageView>,
        shadow_map_matrices_and_splits: &Option<(
            [f32; SHADOW_MAP_CASCADE_COUNT as usize],
            [Mat4; SHADOW_MAP_CASCADE_COUNT as usize],
        )>,
        asset_database: &Arc<RwLock<AssetDatabase>>,
        settings: &Settings,
        g_buffer: &Arc<Framebuffer>,
        world: &World,
    ) -> Result<(), RendererError> {
        self.add_materials_and_textures_to_descriptor_set(context, asset_database)?;

        builder.begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0u32; 2].into()), Some(1f32.into())],
                ..RenderPassBeginInfo::framebuffer(g_buffer.clone())
            },
            SubpassBeginInfo::default(),
        )?;

        if let Some((entities_data, indirect_commands)) =
            self.get_entities_data_and_indirect_draw_commands(context, asset_database)
        {
            self.draw_meshes(
                builder,
                context,
                camera,
                aspect_ratio,
                point_lights_storage_buffer,
                directional_lights_storage_buffer,
                visible_light_indices_storage_buffer,
                lights_from_tile_storage_image_view,
                &entities_data,
                &indirect_commands,
                shadow_map,
                shadow_map_matrices_and_splits,
                asset_database,
                settings,
            )?;
        }

        builder.next_subpass(SubpassEndInfo::default(), SubpassBeginInfo::default())?;

        self.draw_skybox(
            builder,
            context,
            camera,
            aspect_ratio,
            asset_database,
            world,
        )?;

        builder.end_render_pass(Default::default())?;

        Ok(())
    }

    fn draw_meshes(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        context: &RendererContext,
        camera: &Camera,
        aspect_ratio: f32,
        point_lights_storage_buffer: &Option<Subbuffer<[mesh_shaders::fs::PointLight]>>,
        directional_lights_storage_buffer: &Option<Subbuffer<[mesh_shaders::fs::DirectionalLight]>>,
        visible_light_indices_storage_buffer: &Subbuffer<[u32]>,
        lights_from_tile_storage_image_view: &Arc<ImageView>,
        entities_data: &Subbuffer<[mesh_shaders::vs::EntityData]>,
        indirect_commands: &Subbuffer<[DrawIndexedIndirectCommand]>,
        shadow_map: &Arc<ImageView>,
        shadow_map_matrices_and_splits: &Option<(
            [f32; SHADOW_MAP_CASCADE_COUNT as usize],
            [Mat4; SHADOW_MAP_CASCADE_COUNT as usize],
        )>,
        asset_database: &Arc<RwLock<AssetDatabase>>,
        settings: &Settings,
    ) -> Result<(), RendererError> {
        let view = Mat4::look_to_rh(
            camera.position,
            (camera.rotation * Vec3::NEG_Z).normalize(),
            (camera.rotation * Vec3::Y).normalize(),
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
                * Mat4::perspective_rh(camera.fov, aspect_ratio, 0.01, 100.0);

            let cascade_splits = shadow_map_matrices_and_splits
                .map(|tuple| tuple.0)
                .unwrap_or([0.0; SHADOW_MAP_CASCADE_COUNT as usize]);

            let light_space_matrices = shadow_map_matrices_and_splits
                .map(|tuple| tuple.1)
                .unwrap_or([Mat4::IDENTITY; SHADOW_MAP_CASCADE_COUNT as usize]);

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
                cutoff_distances: cascade_splits.map(|split| split.into()),
                light_culling_tile_size: settings.renderer.light_culling.tile_size,
                light_culling_z_slices: settings.renderer.light_culling.z_slices.into(),
                light_culling_sample_count_per_level: settings
                    .renderer
                    .shadow_mapping
                    .sample_count_per_level
                    .map(|x| x.into()),
                shadow_bias: settings.renderer.shadow_mapping.bias.into(),
                shadow_slope_bias: settings.renderer.shadow_mapping.slope_bias.into(),
                shadow_normal_bias: settings.renderer.shadow_mapping.normal_bias.into(),
                penumbra_filter_size: settings.renderer.shadow_mapping.penumbra_max_size.into(),
                light_space_matrices: light_space_matrices.map(|mat| mat.to_cols_array_2d()),
            };

            let buffer = context.uniform_buffer_allocator.allocate_sized().unwrap();
            *buffer.write().unwrap() = uniform_data;

            buffer
        };

        let frame_descriptor_set = DescriptorSet::new(
            context.descriptor_set_allocator.clone(),
            self.mesh_pipeline.layout().set_layouts()[FRAME_DESCRIPTOR_SET].clone(),
            [
                WriteDescriptorSet::buffer(
                    FRAME_DESCRIPTOR_SET_CAMERA_MATRICES_BINDING,
                    frame_uniform_buffer,
                ),
                WriteDescriptorSet::buffer(
                    FRAME_DESCRIPTOR_SET_ENTITY_DATA_BUFFER_BINDING,
                    entities_data.clone(),
                ),
                WriteDescriptorSet::buffer(
                    FRAME_DESCRIPTOR_SET_DIRECTIONAL_LIGHTS_BUFFER_BINDING,
                    directional_lights_storage_buffer
                        .as_ref()
                        .unwrap_or(
                            &context
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
                            &context
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
                WriteDescriptorSet::image_view(
                    FRAME_DESCRIPTOR_SET_SHADOW_MAP_BINDING,
                    shadow_map.clone(),
                ),
            ],
            [],
        )?;

        builder.bind_pipeline_graphics(self.mesh_pipeline.clone())?;

        builder.bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            self.mesh_pipeline.layout().clone(),
            0,
            (
                self.slow_changing_descriptor_set.clone(),
                frame_descriptor_set.clone(),
            ),
        )?;

        let asset_database_read = asset_database.read().unwrap();
        builder
            .bind_vertex_buffers(0, asset_database_read.vertex_buffer().clone())?
            .bind_index_buffer(asset_database_read.index_buffer().clone())?;
        drop(asset_database_read);

        unsafe { builder.draw_indexed_indirect(indirect_commands.clone()) }?;

        Ok(())
    }

    fn draw_skybox(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        context: &RendererContext,
        camera: &Camera,
        aspect_ratio: f32,
        asset_database: &Arc<RwLock<AssetDatabase>>,
        world: &World,
    ) -> Result<(), RendererError> {
        builder.bind_pipeline_graphics(self.skybox_pipeline.clone())?;

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

        let asset_database_read = asset_database.read().unwrap();
        let environment_descriptor_set = DescriptorSet::new(
            context.descriptor_set_allocator.clone(),
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
        )?;
        drop(asset_database_read);

        let frame_uniform_buffer = {
            let proj = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0)) // Correction for vulkan's
                                                                   // inverted y
                * Mat4::perspective_rh(camera.fov, aspect_ratio, 0.01, 100.0);
            let view = Mat4::look_at_rh(
                Vec3::ZERO,
                (camera.rotation * Vec3::NEG_Z).normalize(),
                (camera.rotation * Vec3::Y).normalize(),
            );

            let uniform_data = skybox_shaders::vs::FrameUniforms {
                view_proj: (proj * view).to_cols_array_2d(),
            };

            let buffer = context.uniform_buffer_allocator.allocate_sized().unwrap();
            *buffer.write().unwrap() = uniform_data;

            buffer
        };

        let frame_descriptor_set = DescriptorSet::new(
            context.descriptor_set_allocator.clone(),
            self.skybox_pipeline.layout().set_layouts()[1].clone(),
            [WriteDescriptorSet::buffer(0, frame_uniform_buffer)],
            [],
        )?;

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.skybox_pipeline.layout().clone(),
                0,
                (
                    environment_descriptor_set.clone(),
                    frame_descriptor_set.clone(),
                ),
            )?
            .bind_vertex_buffers(0, self.cube_vertex_buffer.clone())?
            .bind_index_buffer(self.cube_index_buffer.clone())?;

        unsafe { builder.draw_indexed(self.cube_index_buffer.len() as u32, 1, 0, 0, 0) }?;

        Ok(())
    }
}

pub mod mesh_shaders {
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

pub mod skybox_shaders {
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
