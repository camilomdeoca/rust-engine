use std::sync::Arc;

use glam::Vec2;
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo,
    },
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    device::Device,
    format::Format,
    image::sampler::Sampler,
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
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
    render_pass::{Framebuffer, RenderPass, Subpass},
    shader::EntryPoint,
    Validated, VulkanError,
};

use crate::assets::vertex::Vertex;

use super::{Renderer, RendererError};

pub fn create_post_processing_render_pass(
    device: &Arc<Device>,
    output_format: Format,
) -> Result<Arc<RenderPass>, Validated<VulkanError>> {
    vulkano::ordered_passes_renderpass!(
        device.clone(),
        attachments: {
            final_color: {
                format: output_format,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        passes: [
            {
                color: [final_color],
                depth_stencil: {},
                input: []
            },
        ]
    )
}

pub fn create_post_processing_pipeline(
    extent: Vec2,
    sub_pass: Subpass,
    device: &Arc<Device>,
    vs: &EntryPoint,
    fs: &EntryPoint,
    sampler: &Arc<Sampler>,
) -> Result<Arc<GraphicsPipeline>, Validated<VulkanError>> {
    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: extent.into(),
        depth_range: 0.0..=1.0,
    };

    let pipeline_stages = [
        PipelineShaderStageCreateInfo::new(vs.clone()),
        PipelineShaderStageCreateInfo::new(fs.clone()),
    ];
    let mut layout_create_info =
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&pipeline_stages);

    layout_create_info.set_layouts[0]
        .bindings
        .get_mut(&0)
        .unwrap()
        .immutable_samplers = vec![sampler.clone()];

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
            stages: pipeline_stages.into_iter().collect(),
            vertex_input_state: Some(Vertex::per_vertex().definition(vs)?),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState {
                viewports: [viewport.clone()].into_iter().collect(),
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::Back,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                sub_pass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            subpass: Some(sub_pass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
}

impl Renderer {
    pub fn post_processing_pass(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        framebuffer: &Arc<Framebuffer>,
    ) -> Result<(), RendererError> {
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassBeginInfo::default(),
            )?
            .bind_pipeline_graphics(self.post_processing_pipeline.clone())?;
        let descriptor_set = DescriptorSet::new(
            self.context.descriptor_set_allocator.clone(),
            self.post_processing_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view(1, self.g_buffer.attachments()[0].clone()),
                WriteDescriptorSet::image_view(
                    2,
                    self.ambient_occlusion_buffer.attachments()[0].clone(),
                ),
            ],
            [],
        )?;
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.post_processing_pipeline.layout().clone(),
                0,
                descriptor_set,
            )?
            .bind_vertex_buffers(0, self.full_screen_vertex_buffer.clone())?
            .bind_index_buffer(self.full_screen_index_buffer.clone())?;
        unsafe { builder.draw_indexed(self.full_screen_index_buffer.len() as u32, 1, 0, 0, 0) }?;

        builder.end_render_pass(Default::default())?;

        Ok(())
    }
}

pub mod post_processing_shaders {
    pub mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "assets/shaders/post_processing.fs.glsl",
        }
    }
}
