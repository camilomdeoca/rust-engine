use std::sync::Arc;

use glam::{Mat4, Vec2, Vec3};
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo,
    },
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    device::Device,
    format::Format,
    image::{sampler::Sampler, view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage},
    memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator},
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

use crate::{assets::vertex::Vertex, camera::Camera};

use super::{Renderer, RendererError};

pub fn create_ambient_occlusion_framebuffer(
    render_pass: &Arc<RenderPass>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    extent: [u32; 3],
) -> Result<Arc<Framebuffer>, RendererError> {
    let extent = [
        (extent[0] as f32 * 1.0) as u32,
        (extent[1] as f32 * 1.0) as u32,
        extent[2],
    ];

    let ambient_occlusion_buffer = ImageView::new_default(
        Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8_UNORM,
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
            attachments: vec![ambient_occlusion_buffer.clone()],
            ..Default::default()
        },
    )?)
}

pub fn create_ambient_occlusion_render_pass(
    device: &Arc<Device>,
) -> Result<Arc<RenderPass>, Validated<VulkanError>> {
    vulkano::ordered_passes_renderpass!(
        device.clone(),
        attachments: {
            ao_output: {
                format: Format::R8_UNORM,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        passes: [
            // Ambient occlusion
            {
                color: [ao_output],
                depth_stencil: {},
                input: []
            },
        ]
    )
}

pub fn create_ambient_occlusion_pipeline(
    extent: Vec2,
    sub_pass: Subpass,
    device: &Arc<Device>,
    vs: &EntryPoint,
    fs: &EntryPoint,
    sampler_float: &Arc<Sampler>,
    sampler_uint: &Arc<Sampler>,
) -> Result<Arc<GraphicsPipeline>, Validated<VulkanError>> {
    let extent = Vec2::new((extent.x * 1.0).trunc(), (extent.y * 1.0).trunc());
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
        .immutable_samplers = vec![sampler_float.clone()];
    layout_create_info.set_layouts[0]
        .bindings
        .get_mut(&1)
        .unwrap()
        .immutable_samplers = vec![sampler_uint.clone()];

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
    pub fn ambient_occlusion_pass(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        camera: &Camera,
        aspect_ratio: f32,
    ) -> Result<(), RendererError> {
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(self.ambient_occlusion_buffer.clone())
                },
                SubpassBeginInfo::default(),
            )?
            .bind_pipeline_graphics(self.ambient_occlusion_pipeline.clone())?;

        let uniform_buffer = {
            let view = Mat4::look_to_rh(
                camera.position,
                (camera.rotation * Vec3::NEG_Z).normalize(),
                (camera.rotation * Vec3::Y).normalize(),
            );
            let proj = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0))
                * Mat4::perspective_rh(camera.fov, aspect_ratio, 0.01, 100.0);

            let uniform_data = ambient_occlusion_shaders::fs::Uniforms {
                inverse_projection: proj.inverse().to_cols_array_2d(),
                view: view.to_cols_array_2d(),
                projection: proj.to_cols_array_2d(),
                sample_count: self.settings.sample_count as f32,
                sample_radius: self.settings.sample_radius,
                intensity: self.settings.ambient_occlusion_intensity,
            };

            let buffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
            *buffer.write().unwrap() = uniform_data;

            buffer
        };

        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.ambient_occlusion_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view(2, self.g_buffer.attachments()[0].clone()),
                WriteDescriptorSet::image_view(3, self.g_buffer.attachments()[1].clone()),
                WriteDescriptorSet::buffer(4, uniform_buffer.clone()),
            ],
            [],
        )?;
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.ambient_occlusion_pipeline.layout().clone(),
                0,
                descriptor_set,
            )?
            .bind_vertex_buffers(0, self.full_screen_vertex_buffer.clone())?
            .bind_index_buffer(self.full_screen_index_buffer.clone())?;
        assert_eq!(self.full_screen_index_buffer.len(), 6);
        unsafe { builder.draw_indexed(self.full_screen_index_buffer.len() as u32, 1, 0, 0, 0) }?;

        builder.end_render_pass(Default::default())?;

        Ok(())
    }
}

pub mod ambient_occlusion_shaders {
    pub mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "assets/shaders/ambient_occlusion.fs.glsl",
        }
    }
}
