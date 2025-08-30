use std::sync::Arc;

use flecs_ecs::prelude::*;
use glam::{Mat4, Vec2, Vec3};
use vulkano::{
    buffer::Subbuffer, command_buffer::{AutoCommandBufferBuilder, DrawIndexedIndirectCommand, PrimaryAutoCommandBuffer}, descriptor_set::{layout::{DescriptorBindingFlags, DescriptorType}, DescriptorSet, WriteDescriptorSet}, device::Device, format::Format, image::{
        sampler::Sampler, view::{ImageView, ImageViewCreateInfo}, Image, ImageCreateInfo, ImageSubresourceRange, ImageType, ImageUsage
    }, memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator}, pipeline::{graphics::{color_blend::{ColorBlendAttachmentState, ColorBlendState}, depth_stencil::{CompareOp, DepthState, DepthStencilState}, input_assembly::InputAssemblyState, multisample::MultisampleState, rasterization::{CullMode, RasterizationState}, vertex_input::{Vertex as _, VertexDefinition}, viewport::{Viewport, ViewportState}, GraphicsPipelineCreateInfo}, layout::PipelineDescriptorSetLayoutCreateInfo, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo}, render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass}, shader::EntryPoint, Validated, VulkanError
};

use crate::{assets::vertex::Vertex, camera::Camera, ecs::components};

use super::{Renderer, RendererError, SHADOW_MAP_CASCADE_COUNT};

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

pub fn create_g_buffer_framebuffer(
    render_pass: &Arc<RenderPass>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    extent: [u32; 3],
) -> Result<Arc<Framebuffer>, RendererError> {
    // We need to build mip chain for rendering SSAO at lower resolution and to then upscale it
    // with Joint Bilateral upsampling. We will also use it to sample smaller levels first:
    // We will store the min of 4 pixels in each next mip level, and if when comparing the depth it
    // is outside the radius then we dont sample bigger level as all will be bigger (they are
    // behind)
    let mip_levels = extent.iter().max().unwrap().ilog2();
    let depth_buffer_image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::D32_SFLOAT,
            extent,
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
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R32G32_UINT,
                extent,
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap(),
    )?;

    Ok(Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![g_buffer.clone(), depth_buffer.clone()],
            ..Default::default()
        },
    )?)
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

impl Renderer {
    pub fn draw_meshes(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        camera: &Camera,
        aspect_ratio: f32,
        point_lights_storage_buffer: &Option<Subbuffer<[mesh_shaders::fs::PointLight]>>,
        directional_lights_storage_buffer: &Option<Subbuffer<[mesh_shaders::fs::DirectionalLight]>>,
        visible_light_indices_storage_buffer: &Subbuffer<[u32]>,
        lights_from_tile_storage_image_view: &Arc<ImageView>,
        entities_data: &Subbuffer<[mesh_shaders::vs::EntityData]>,
        indirect_commands: &Subbuffer<[DrawIndexedIndirectCommand]>,
        shadow_map_matrices_and_splits: &Option<(
            [f32; SHADOW_MAP_CASCADE_COUNT as usize],
            [Mat4; SHADOW_MAP_CASCADE_COUNT as usize],
        )>,
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
                light_culling_tile_size: self.settings.light_culling_tile_size,
                light_culling_z_slices: self.settings.light_culling_z_slices.into(),
                light_culling_sample_count_per_level: self
                    .settings
                    .sample_count_per_level
                    .map(|x| x.into()),
                shadow_bias: self.settings.shadow_bias.into(),
                shadow_slope_bias: self.settings.shadow_slope_bias.into(),
                shadow_normal_bias: self.settings.shadow_normal_bias.into(),
                penumbra_filter_size: self.settings.penumbra_max_size.into(),
                light_space_matrices: light_space_matrices.map(|mat| mat.to_cols_array_2d()),
            };

            let buffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
            *buffer.write().unwrap() = uniform_data;

            buffer
        };

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
                    entities_data.clone(),
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
                WriteDescriptorSet::image_view(
                    FRAME_DESCRIPTOR_SET_SHADOW_MAP_BINDING,
                    self.shadow_map_framebuffer.attachments()[0].clone(),
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

        let asset_database_read = self.asset_database.read().unwrap();
        builder
            .bind_vertex_buffers(0, asset_database_read.vertex_buffer().clone())?
            .bind_index_buffer(asset_database_read.index_buffer().clone())?;
        drop(asset_database_read);

        unsafe { builder.draw_indexed_indirect(indirect_commands.clone()) }?;

        Ok(())
    }

    pub fn draw_skybox(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        camera: &Camera,
        aspect_ratio: f32,
    ) -> Result<(), RendererError> {
        builder.bind_pipeline_graphics(self.skybox_pipeline.clone())?;

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

            let buffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
            *buffer.write().unwrap() = uniform_data;

            buffer
        };

        let frame_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
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
