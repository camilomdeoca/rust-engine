use std::sync::Arc;

use flecs_ecs::{
    core::World,
    prelude::{Builder, QueryAPI, QueryBuilderImpl},
};
use glam::{Mat3, Mat4, Vec2, Vec3, Vec4Swizzles};
use thiserror::Error;
use vulkano::{
    buffer::Subbuffer, command_buffer::{
        AutoCommandBufferBuilder, DrawIndexedIndirectCommand, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    }, descriptor_set::{DescriptorSet, WriteDescriptorSet}, format::Format, image::{view::ImageView, Image, ImageCreateInfo, ImageLayout, ImageType, ImageUsage}, memory::allocator::AllocationCreateInfo, pipeline::{
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
    }, render_pass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, Framebuffer, FramebufferCreateInfo, RenderPass, RenderPassCreateInfo, Subpass, SubpassDescription
    }, Validated, ValidationError, VulkanError
};

use crate::{
    assets::vertex::Vertex,
    camera::Camera,
    ecs::components::{DirectionalLight, DirectionalLightShadowMap, Transform},
    settings::{Settings, SHADOW_MAP_CASCADE_COUNT},
};

use super::{meshes_pass::mesh_shaders, RendererContext};

#[derive(Error, Debug)]
pub enum CascadedShadowMapsPassError {
    #[error("{0}")]
    VulkanoVulkanError(#[from] Validated<VulkanError>),
    #[error("{0}")]
    VulkanoValidationError(#[from] Box<ValidationError>),
}

pub struct CascadedShadowMapsPass {
    pipeline: Arc<GraphicsPipeline>,
    render_pass: Arc<RenderPass>,
}

fn compute_shadow_map_matrices_and_splits(
    camera: &Camera,
    aspect_ratio: f32,
    world: &World,
    settings: &Settings,
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
        let d = settings.renderer.shadow_mapping.cascade_split_lambda * (log - uniform) + uniform;
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

    let query = world
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

            let shadow_map_size = settings.renderer.shadow_mapping.cascade_level_size;
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

impl CascadedShadowMapsPass {
    pub fn new(
        context: &RendererContext,
        settings: &Settings,
    ) -> Result<Self, CascadedShadowMapsPassError> {
        let render_pass = RenderPass::new(
            context.device.clone(),
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
        )?;

        let vs = vs::load(context.device.clone())?
            .entry_point("main")
            .unwrap();
        let fs = fs::load(context.device.clone())?
            .entry_point("main")
            .unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(vs.clone()),
            PipelineShaderStageCreateInfo::new(fs.clone()),
        ];

        let layout_create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);

        let layout = PipelineLayout::new(
            context.device.clone(),
            layout_create_info
                .into_pipeline_layout_create_info(context.device.clone())
                .unwrap(),
        )?;

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [settings.renderer.shadow_mapping.cascade_level_size as f32; 2],
            depth_range: 0.0..=1.0,
        };

        let pipeline = GraphicsPipeline::new(
            context.device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(Vertex::per_vertex().definition(&vs)?),
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
        )?;

        Ok(Self {
            pipeline,
            render_pass,
        })
    }

    pub fn create_framebuffer(
        &self,
        context: &RendererContext,
        size: u32,
    ) -> Result<Arc<Framebuffer>, CascadedShadowMapsPassError> {
        let shadow_map = ImageView::new_default(
            Image::new(
                context.memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: self.render_pass.attachments()[0].format,
                    extent: [size, size, 1],
                    usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::SAMPLED,
                    array_layers: SHADOW_MAP_CASCADE_COUNT,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )?;

        Ok(Framebuffer::new(
            self.render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![shadow_map.clone()],
                ..Default::default()
            },
        )?)
    }

    pub fn execute(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        context: &RendererContext,
        shadow_map_framebuffer: &Arc<Framebuffer>,
        entities_data: &Subbuffer<[mesh_shaders::vs::EntityData]>, // In the future will be
        indirect_commands: &Subbuffer<[DrawIndexedIndirectCommand]>, // computed here to cull
        // entities outside shadow
        // mapping range
        vertex_buffer: &Subbuffer<[Vertex]>,
        index_buffer: &Subbuffer<[u32]>,
        camera: &Camera,
        aspect_ratio: f32,
        world: &World,
        settings: &Settings,
    ) -> Result<
        (
            Arc<ImageView>,
            Option<(
                [f32; SHADOW_MAP_CASCADE_COUNT as usize],
                [Mat4; SHADOW_MAP_CASCADE_COUNT as usize],
            )>,
        ),
        CascadedShadowMapsPassError,
    > {
        let shadow_map_matrices_and_splits =
            compute_shadow_map_matrices_and_splits(&camera, aspect_ratio, world, settings);
        if shadow_map_matrices_and_splits.is_none() {
            return Ok((shadow_map_framebuffer.attachments()[0].clone(), None));
        }

        let (depth_splits, light_space_matrices) = shadow_map_matrices_and_splits.unwrap();
        builder.begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some(1f32.into())],

                ..RenderPassBeginInfo::framebuffer(shadow_map_framebuffer.clone())
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )?;

        let light_data_buffer = context.uniform_buffer_allocator.allocate_sized().unwrap();
        *light_data_buffer.write().unwrap() = vs::LightData {
            light_space_matrices: light_space_matrices.map(|mat| mat.to_cols_array_2d()),
        };

        let descriptor_set = DescriptorSet::new(
            context.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, light_data_buffer),
                WriteDescriptorSet::buffer(1, entities_data.clone()),
            ],
            [],
        )?;

        builder
            .bind_pipeline_graphics(self.pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                descriptor_set.clone(),
            )?;

        builder
            .bind_vertex_buffers(0, vertex_buffer.clone())?
            .bind_index_buffer(index_buffer.clone())?;

        unsafe { builder.draw_indexed_indirect(indirect_commands.clone()) }?;

        builder.end_render_pass(Default::default())?;

        Ok((
            shadow_map_framebuffer.attachments()[0].clone(),
            Some((depth_splits, light_space_matrices)),
        ))
    }
}

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
