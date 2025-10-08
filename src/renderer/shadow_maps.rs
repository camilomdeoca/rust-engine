use std::sync::Arc;

use flecs_ecs::prelude::*;
use glam::{Mat3, Mat4, Vec2, Vec3, Vec4Swizzles};
use vulkano::{
    buffer::Subbuffer,
    command_buffer::{
        AutoCommandBufferBuilder, DrawIndexedIndirectCommand, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    },
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    device::{Device, DeviceOwned},
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageLayout, ImageType, ImageUsage},
    memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            depth_stencil::{DepthState, DepthStencilState},
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
    render_pass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp,
        Framebuffer, FramebufferCreateInfo, RenderPass, RenderPassCreateInfo, Subpass,
        SubpassDescription,
    },
    shader::EntryPoint,
    Validated, VulkanError,
};

use crate::{
    assets::vertex::Vertex,
    camera::Camera,
    ecs::components::{DirectionalLight, DirectionalLightShadowMap, Transform},
};

use super::{main_pass::mesh_shaders, Renderer, RendererError};

pub const SHADOW_MAP_CASCADE_COUNT: u32 = 4;

pub struct ShadowMappingSettings {
    pub cascade_level_size: u32,
    pub sample_count_per_level: [u32; SHADOW_MAP_CASCADE_COUNT as usize],
    pub bias: f32,
    pub slope_bias: f32,
    pub normal_bias: f32,
    pub penumbra_max_size: f32,
    pub cascade_split_lambda: f32,
}

pub fn create_shadow_map_renderpass(
    device: &Arc<Device>,
) -> Result<Arc<RenderPass>, Validated<VulkanError>> {
    RenderPass::new(
        device.clone(),
        RenderPassCreateInfo {
            attachments: [AttachmentDescription {
                format: Format::D32_SFLOAT,
                load_op: AttachmentLoadOp::Clear,
                store_op: AttachmentStoreOp::Store,
                initial_layout: ImageLayout::DepthStencilAttachmentOptimal,
                final_layout: ImageLayout::DepthStencilAttachmentOptimal,
                ..Default::default()
            }]
            .into(),
            subpasses: [SubpassDescription {
                view_mask: (1 << SHADOW_MAP_CASCADE_COUNT) - 1,
                depth_stencil_attachment: Some(AttachmentReference {
                    attachment: 0,
                    layout: ImageLayout::DepthStencilAttachmentOptimal,
                    ..Default::default()
                }),
                ..Default::default()
            }]
            .into(),
            correlated_view_masks: [(1 << SHADOW_MAP_CASCADE_COUNT) - 1].into(),
            ..Default::default()
        },
    )
}

pub fn create_shadow_map_pipeline(
    extent: Vec2,
    render_pass: &Arc<RenderPass>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    shadow_mapping_vs: &EntryPoint,
    shadow_mapping_fs: &EntryPoint,
) -> Result<Arc<GraphicsPipeline>, Validated<VulkanError>> {
    let device = memory_allocator.device();
    let stages = [
        PipelineShaderStageCreateInfo::new(shadow_mapping_vs.clone()),
        PipelineShaderStageCreateInfo::new(shadow_mapping_fs.clone()),
    ];
    let layout_create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);

    let layout = PipelineLayout::new(
        device.clone(),
        layout_create_info
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )?;

    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: extent.into(),
        depth_range: 0.0..=1.0,
    };

    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(Vertex::per_vertex().definition(shadow_mapping_vs)?),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState {
                viewports: [viewport.clone()].into(),
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
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
}

pub fn create_shadow_map_framebuffer(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    shadow_mapping_render_pass: &Arc<RenderPass>,
    size: u32,
) -> Result<Arc<Framebuffer>, Validated<VulkanError>> {
    let shadow_map = ImageView::new_default(
        Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: shadow_mapping_render_pass.attachments()[0].format,
                extent: [size, size, 1],
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::SAMPLED,
                array_layers: SHADOW_MAP_CASCADE_COUNT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap(),
    )?;

    Framebuffer::new(
        shadow_mapping_render_pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![shadow_map.clone()],
            ..Default::default()
        },
    )
}

impl Renderer {
    pub fn compute_shadow_map_matrices_and_splits(
        &mut self,
        camera: &Camera,
        aspect_ratio: f32,
    ) -> Option<(
        [f32; SHADOW_MAP_CASCADE_COUNT as usize],
        [Mat4; SHADOW_MAP_CASCADE_COUNT as usize],
    )> {
        let near_clip = 0.01;
        let far_clip = 100.0;
        let clip_range = far_clip - near_clip;

        let min_z = near_clip;
        let max_z = near_clip + clip_range;

        let range = max_z - min_z;
        let ratio: f32 = max_z / min_z;

        let cascade_splits: [f32; SHADOW_MAP_CASCADE_COUNT as usize] = std::array::from_fn(|i| {
            let p = (i + 1) as f32 / SHADOW_MAP_CASCADE_COUNT as f32;
            let log = min_z * ratio.powf(p);
            let uniform = min_z + range * p;
            let d = self.settings.shadow_map_cascade_split_lambda * (log - uniform) + uniform;
            (d - near_clip) / clip_range
        });

        const NDC_CORNERS: [Vec3; 8] = [
            Vec3::new(-1.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(-1.0, -1.0, 1.0),
        ];

        let camera_view = Mat4::look_to_rh(
            camera.position,
            (camera.rotation * Vec3::NEG_Z).normalize(),
            (camera.rotation * Vec3::Y).normalize(),
        );
        let camera_projection = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0))
            * Mat4::perspective_rh(camera.fov, aspect_ratio, 0.01, 100.0);

        let inverse_camera_proj = camera_projection.inverse();

        let full_frustum_corners = NDC_CORNERS.map(|ndc_corner| {
            let corner = inverse_camera_proj * ndc_corner.extend(1.0);
            corner.xyz() / corner.w
        });

        let mut light_space_matrices = [Mat4::IDENTITY; SHADOW_MAP_CASCADE_COUNT as usize];
        let mut split_depths = [0.0; SHADOW_MAP_CASCADE_COUNT as usize];

        let query = self
            .world
            .query::<&Transform>()
            .with::<DirectionalLight>()
            .with::<DirectionalLightShadowMap>()
            .build();

        let mut count = 0;
        query.each(|transform| {
            count += 1;

            let mut last_split_dist = 0.0;
            for i in 0..SHADOW_MAP_CASCADE_COUNT as usize {
                let split_dist = cascade_splits[i];
                let mut frustum_corners = full_frustum_corners;

                for j in 0..4 {
                    let dist = frustum_corners[j + 4] + frustum_corners[j];
                    frustum_corners[j + 4] = frustum_corners[j] + (dist * split_dist);
                    frustum_corners[j] = frustum_corners[j] + (dist * last_split_dist);
                }

                let mut frustum_center = frustum_corners
                    .iter()
                    .cloned()
                    .fold(Vec3::ZERO, |acc, e| acc + e)
                    / 8.0;

                let mut max_diagonal: f32 = 0.0;
                for p1 in frustum_corners {
                    for p2 in frustum_corners {
                        max_diagonal = max_diagonal.max(p1.distance(p2));
                    }
                }

                let light_direction = (transform.rotation * Vec3::NEG_Z).normalize();

                let shadow_map_size = self.shadow_map_framebuffer.extent()[0];
                assert_eq!(shadow_map_size, self.shadow_map_framebuffer.extent()[1]);
                let pixel_size = max_diagonal / shadow_map_size as f32;

                frustum_center = {
                    let frustum_center_vec4 = camera_view.inverse() * frustum_center.extend(1.0);
                    frustum_center_vec4.xyz() / frustum_center_vec4.w
                };

                // Align frustum_center to pixels so it doesnt move
                let light_rotation_matrix = Mat3::look_to_rh(light_direction, Vec3::Y);
                frustum_center = light_rotation_matrix * frustum_center;
                frustum_center.x = (frustum_center.x / pixel_size).round() * pixel_size;
                frustum_center.y = (frustum_center.y / pixel_size).round() * pixel_size;
                frustum_center = light_rotation_matrix.inverse() * frustum_center;

                let max_extents = Vec2::splat(max_diagonal / 2.0);
                let min_extents = -max_extents;

                let light_projection = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0))
                    * Mat4::orthographic_rh(
                        min_extents.x,
                        max_extents.x,
                        min_extents.y,
                        max_extents.y,
                        -50.0,
                        50.0,
                    );

                let light_view = Mat4::look_to_rh(
                    frustum_center, // TODO: Use camera pos?
                    light_direction,
                    Vec3::Y,
                );

                split_depths[i] = (near_clip + split_dist * clip_range) * -1.0;
                light_space_matrices[i] = light_projection * light_view;
                last_split_dist = split_dist;
            }
        });

        assert!(
            count == 0 || count == 1,
            "We support at most one light with shadow maps"
        );

        if count == 0 {
            None
        } else {
            Some((split_depths, light_space_matrices))
        }
    }

    pub fn render_shadow_maps(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        entities_data: &Subbuffer<[mesh_shaders::vs::EntityData]>,
        indirect_commands: &Subbuffer<[DrawIndexedIndirectCommand]>,
        light_space_matrices: &[Mat4; SHADOW_MAP_CASCADE_COUNT as usize],
    ) -> Result<(), RendererError> {
        builder.begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some(1f32.into())],

                ..RenderPassBeginInfo::framebuffer(self.shadow_map_framebuffer.clone())
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )?;

        let light_data_buffer = self.context.uniform_buffer_allocator.allocate_sized().unwrap();
        *light_data_buffer.write().unwrap() = shadow_mapping_shaders::vs::LightData {
            light_space_matrices: light_space_matrices.map(|mat| mat.to_cols_array_2d()),
        };

        let descriptor_set = DescriptorSet::new(
            self.context.descriptor_set_allocator.clone(),
            self.shadow_mapping_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, light_data_buffer),
                WriteDescriptorSet::buffer(1, entities_data.clone()),
            ],
            [],
        )?;

        builder
            .bind_pipeline_graphics(self.shadow_mapping_pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.shadow_mapping_pipeline.layout().clone(),
                0,
                descriptor_set.clone(),
            )?;

        let vertex_buffer;
        let index_buffer;
        {
            let asset_database_read = self.asset_database.read().unwrap();
            vertex_buffer = asset_database_read.vertex_buffer().clone();
            index_buffer = asset_database_read.index_buffer().clone();
        }

        builder
            .bind_vertex_buffers(0, vertex_buffer.clone())?
            .bind_index_buffer(index_buffer.clone())?;

        unsafe { builder.draw_indexed_indirect(indirect_commands.clone()) }?;

        builder.end_render_pass(Default::default())?;

        Ok(())
    }
}

pub mod shadow_mapping_shaders {
    pub mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "assets/shaders/shadow_mapping.vs.glsl",
        }
    }

    pub mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "assets/shaders/shadow_mapping.fs.glsl",
        }
    }
}
