use std::sync::Arc;

use glam::UVec2;
use vulkano::{
    buffer::Subbuffer,
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo,
    },
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    device::Device,
    format::Format,
    image::{
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, LOD_CLAMP_NONE},
        view::ImageView,
    },
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
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::EntryPoint,
    Validated, VulkanError,
};

use crate::assets::vertex::Vertex;

use super::{full_screen_quad, RendererContext, RendererError};

pub struct PostProcessingPass {
    pipeline: Arc<GraphicsPipeline>,
    render_pass: Arc<RenderPass>,
}

fn create_post_processing_render_pass(
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

fn create_post_processing_pipeline(
    extent: UVec2,
    sub_pass: Subpass,
    device: &Arc<Device>,
    vs: &EntryPoint,
    fs: &EntryPoint,
    sampler: &Arc<Sampler>,
) -> Result<Arc<GraphicsPipeline>, Validated<VulkanError>> {
    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: extent.as_vec2().into(),
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

impl PostProcessingPass {
    pub fn new(
        context: &RendererContext,
        output_extent: UVec2,
        output_format: Format,
    ) -> Result<Self, RendererError> {
        let full_screen_quad_vs = full_screen_quad::vs::load(context.device.clone())?
            .entry_point("main")
            .unwrap();

        let fs = post_processing_shaders::fs::load(context.device.clone())?
            .entry_point("main")
            .unwrap();

        let render_pass = create_post_processing_render_pass(&context.device, output_format)?;

        let sampler = Sampler::new(
            context.device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                lod: 0.0..=LOD_CLAMP_NONE,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )?;

        let pipeline = create_post_processing_pipeline(
            output_extent,
            Subpass::from(render_pass.clone(), 0).unwrap(),
            &context.device,
            &full_screen_quad_vs,
            &fs,
            &sampler,
        )?;

        Ok(Self {
            pipeline,
            render_pass,
        })
    }

    pub fn create_framebuffers(
        &self,
        image_views: &[Arc<ImageView>],
    ) -> Result<Vec<Arc<Framebuffer>>, RendererError> {
        let framebuffers = image_views
            .iter()
            .map(|image_view| {
                Framebuffer::new(
                    self.render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![image_view.clone()],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        Ok(framebuffers)
    }

    pub fn execute(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        context: &RendererContext,
        framebuffer: &Arc<Framebuffer>,
        g_buffer: &Arc<ImageView>,
        ambient_occlusion_buffer: &Arc<ImageView>,
        full_screen_vertex_buffer: &Subbuffer<[Vertex]>,
        full_screen_index_buffer: &Subbuffer<[u32]>,
    ) -> Result<(), RendererError> {
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassBeginInfo::default(),
            )?
            .bind_pipeline_graphics(self.pipeline.clone())?;
        let descriptor_set = DescriptorSet::new(
            context.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view(1, g_buffer.clone()),
                WriteDescriptorSet::image_view(2, ambient_occlusion_buffer.clone()),
            ],
            [],
        )?;
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                descriptor_set,
            )?
            .bind_vertex_buffers(0, full_screen_vertex_buffer.clone())?
            .bind_index_buffer(full_screen_index_buffer.clone())?;
        unsafe { builder.draw_indexed(full_screen_index_buffer.len() as u32, 1, 0, 0, 0) }?;

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
