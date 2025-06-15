use std::sync::Arc;

use glam::{Mat4, Vec3};
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        BufferUsage, Subbuffer,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, RenderPassBeginInfo
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, layout::DescriptorType, DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet
    },
    device::{Device, Queue},
    format::Format,
    image::{
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::{ImageView, ImageViewCreateInfo, ImageViewType},
        Image, ImageSubresourceRange,
    },
    memory::allocator::{MemoryTypeFilter, StandardMemoryAllocator},
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
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::GpuFuture, DeviceSize,
};

use crate::assets::{loaders::mesh_loader::{load_mesh_from_buffers, load_mesh_from_buffers_into_new_buffers}, vertex::Vertex};

pub struct ImageBasedLightingMapsGenerator {
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    uniform_buffer_allocator: SubbufferAllocator,
    irradiance_map_generation_render_pass: Arc<RenderPass>,
    irradiance_map_generation_pipeline: Arc<GraphicsPipeline>,
    prefiltered_environment_map_generation_render_pass: Arc<RenderPass>,
    prefiltered_environment_map_generation_pipeline: Arc<GraphicsPipeline>,
    environment_brdf_lut_generation_render_pass: Arc<RenderPass>,
    environment_brdf_lut_generation_pipeline: Arc<GraphicsPipeline>,
    cube_vertex_buffer: Subbuffer<[Vertex]>,
    cube_index_buffer: Subbuffer<[u32]>,
    quad_vertex_buffer: Subbuffer<[Vertex]>,
    quad_index_buffer: Subbuffer<[u32]>,
}

fn create_irradiance_map_generation_pipeline(
    device: Arc<Device>,
    dst_image_format: Format,
) -> (Arc<RenderPass>, Arc<GraphicsPipeline>) {
    let render_pass = vulkano::ordered_passes_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: dst_image_format,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        passes: [
            {
                color: [color],
                depth_stencil: {},
                input: [],
            },
        ],
    )
    .unwrap();

    let vs = cubemap_vs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let fs = irradiance_map_generation_fs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();

    let mesh_pipeline_stages = [
        PipelineShaderStageCreateInfo::new(vs.clone()),
        PipelineShaderStageCreateInfo::new(fs.clone()),
    ];
    let mut layout_create_info =
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&mesh_pipeline_stages);

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
        .descriptor_type = DescriptorType::UniformBufferDynamic;

    let layout = PipelineLayout::new(
        device.clone(),
        layout_create_info
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();

    let sub_pass = Subpass::from(render_pass.clone(), 0).unwrap();

    let pipeline = GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: mesh_pipeline_stages.into_iter().collect(),
            vertex_input_state: Some(Vertex::per_vertex().definition(&vs).unwrap()),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::None,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                sub_pass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(sub_pass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap();

    (render_pass, pipeline)
}

fn create_environment_brdf_lut_generation_pipeline(
    device: Arc<Device>,
) -> (Arc<RenderPass>, Arc<GraphicsPipeline>) {
    let render_pass = vulkano::ordered_passes_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: Format::R16G16_SFLOAT,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        passes: [
            {
                color: [color],
                depth_stencil: {},
                input: [],
            },
        ],
    )
    .unwrap();

    let vs = environment_brdf_lut_generation::vs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let fs = environment_brdf_lut_generation::fs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();

    let mesh_pipeline_stages = [
        PipelineShaderStageCreateInfo::new(vs.clone()),
        PipelineShaderStageCreateInfo::new(fs.clone()),
    ];
    let layout_create_info =
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&mesh_pipeline_stages);

    let layout = PipelineLayout::new(
        device.clone(),
        layout_create_info
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();

    let sub_pass = Subpass::from(render_pass.clone(), 0).unwrap();

    let pipeline = GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: mesh_pipeline_stages.into_iter().collect(),
            vertex_input_state: Some(Vertex::per_vertex().definition(&vs).unwrap()),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::None,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                sub_pass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(sub_pass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap();

    (render_pass, pipeline)
}

fn create_prefiltered_environment_map_generation_pipeline(
    device: Arc<Device>,
    dst_image_format: Format,
) -> (Arc<RenderPass>, Arc<GraphicsPipeline>) {
    let render_pass = vulkano::ordered_passes_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: dst_image_format,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        passes: [
            {
                color: [color],
                depth_stencil: {},
                input: [],
            },
        ],
    )
    .unwrap();

    let vs = cubemap_vs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let fs = prefiltered_environment_map_generation_fs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();

    let mesh_pipeline_stages = [
        PipelineShaderStageCreateInfo::new(vs.clone()),
        PipelineShaderStageCreateInfo::new(fs.clone()),
    ];
    let mut layout_create_info =
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&mesh_pipeline_stages);

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
        .descriptor_type = DescriptorType::UniformBufferDynamic;

    let layout = PipelineLayout::new(
        device.clone(),
        layout_create_info
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();

    let sub_pass = Subpass::from(render_pass.clone(), 0).unwrap();

    let pipeline = GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: mesh_pipeline_stages.into_iter().collect(),
            vertex_input_state: Some(Vertex::per_vertex().definition(&vs).unwrap()),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::None,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                sub_pass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(sub_pass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap();

    (render_pass, pipeline)
}

impl ImageBasedLightingMapsGenerator {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        dst_image_format: Format,
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

        let (cube_vertex_buffer, cube_index_buffer) = load_mesh_from_buffers_into_new_buffers(
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            queue.clone(),
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

        let (quad_vertex_buffer, quad_index_buffer) = load_mesh_from_buffers_into_new_buffers(
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            queue.clone(),
            vec![
                Vertex {
                    a_position: [-1.0, -1.0, 0.0],
                    a_normal: [0.0; 3],
                    a_tangent: [0.0; 3],
                    a_uv: [0.0, 0.0],
                },
                Vertex {
                    a_position: [1.0, -1.0, 0.0],
                    a_normal: [0.0; 3],
                    a_tangent: [0.0; 3],
                    a_uv: [1.0, 0.0],
                },
                Vertex {
                    a_position: [1.0, 1.0, 0.0],
                    a_normal: [0.0; 3],
                    a_tangent: [0.0; 3],
                    a_uv: [1.0, 1.0],
                },
                Vertex {
                    a_position: [-1.0, 1.0, 0.0],
                    a_normal: [0.0; 3],
                    a_tangent: [0.0; 3],
                    a_uv: [0.0, 1.0],
                },
            ],
            vec![0, 1, 2, 2, 3, 0],
        )
        .unwrap();

        let (irradiance_map_generation_render_pass, irradiance_map_generation_pipeline) =
            create_irradiance_map_generation_pipeline(device.clone(), dst_image_format);
        let (
            prefiltered_environment_map_generation_render_pass,
            prefiltered_environment_map_generation_pipeline,
        ) = create_prefiltered_environment_map_generation_pipeline(
            device.clone(),
            dst_image_format,
        );
        let (environment_brdf_lut_generation_render_pass, environment_brdf_lut_generation_pipeline) =
            create_environment_brdf_lut_generation_pipeline(device.clone());

        Self {
            device,
            queue,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            uniform_buffer_allocator,
            irradiance_map_generation_render_pass,
            irradiance_map_generation_pipeline,
            prefiltered_environment_map_generation_render_pass,
            prefiltered_environment_map_generation_pipeline,
            environment_brdf_lut_generation_render_pass,
            environment_brdf_lut_generation_pipeline,
            cube_vertex_buffer,
            cube_index_buffer,
            quad_vertex_buffer,
            quad_index_buffer,
        }
    }

    fn generate_maps(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        environment_map: Arc<ImageView>,
        irradiance_map_image: Arc<Image>,
    ) {
        let proj = Mat4::perspective_rh_gl(90f32.to_radians(), 1.0, 0.01, 10.0);
        let view_projs = [
            proj * Mat4::look_at_rh(Vec3::ZERO, Vec3::X, Vec3::NEG_Y),
            proj * Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_X, Vec3::NEG_Y),
            proj * Mat4::look_at_rh(Vec3::ZERO, Vec3::Y, Vec3::Z),
            proj * Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Y, Vec3::NEG_Z),
            proj * Mat4::look_at_rh(Vec3::ZERO, Vec3::Z, Vec3::NEG_Y),
            proj * Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::NEG_Y),
        ];

        let environment_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.irradiance_map_generation_pipeline
                .layout()
                .set_layouts()[0]
                .clone(),
            [WriteDescriptorSet::image_view(1, environment_map.clone())],
            [],
        )
        .unwrap();
        
        let min_dynamic_align = self
            .device
            .physical_device()
            .properties()
            .min_uniform_buffer_offset_alignment
            .as_devicesize();

        let align =
            (size_of::<cubemap_vs::FrameUniforms>() as DeviceSize + min_dynamic_align - 1)
                & !(min_dynamic_align - 1);

        let uniform_buffer = self.uniform_buffer_allocator.allocate_slice(6 * align).unwrap();
        for (view_proj_index, view_proj) in view_projs.iter().enumerate() {
            *uniform_buffer
                .clone()
                .slice(view_proj_index as DeviceSize * align..)
                .cast_aligned()
                .index(0)
                .write()
                .unwrap() = cubemap_vs::FrameUniforms {
                    view_proj: view_proj.to_cols_array_2d(),
                };
        }

        let frame_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.irradiance_map_generation_pipeline
                .layout()
                .set_layouts()[1]
                .clone(),
            [WriteDescriptorSet::buffer_with_range(
                0,
                DescriptorBufferInfo {
                    buffer: uniform_buffer.clone(),
                    range: 0..size_of::<cubemap_vs::FrameUniforms>() as DeviceSize,
                },
            )],
            [],
        )
        .unwrap();

        for view_proj_index in 0..view_projs.len() {
            let image_view_create_info = ImageViewCreateInfo {
                view_type: ImageViewType::Dim2d,
                subresource_range: ImageSubresourceRange {
                    array_layers: view_proj_index as u32..view_proj_index as u32 + 1,
                    ..irradiance_map_image.subresource_range()
                },
                ..ImageViewCreateInfo::from_image(&irradiance_map_image)
            };
            let image_view = ImageView::new(irradiance_map_image.clone(), image_view_create_info).unwrap();

            let framebuffer = Framebuffer::new(
                self.irradiance_map_generation_render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![image_view],
                    ..Default::default()
                },
            )
            .unwrap();

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    Default::default(),
                )
                .unwrap()
                .set_viewport(
                    0,
                    [Viewport {
                        offset: [0.0, 0.0],
                        extent: [
                            framebuffer.extent()[0] as f32,
                            framebuffer.extent()[1] as f32,
                        ],
                        depth_range: 0.0..=1.0,
                    }]
                    .into_iter()
                    .collect(),
                )
                .unwrap()
                .bind_pipeline_graphics(self.irradiance_map_generation_pipeline.clone())
                .unwrap();

            builder
                .bind_vertex_buffers(0, self.cube_vertex_buffer.clone())
                .unwrap()
                .bind_index_buffer(self.cube_index_buffer.clone())
                .unwrap();

            let offset = (view_proj_index as DeviceSize * align) as u32;
            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.irradiance_map_generation_pipeline.layout().clone(),
                    0,
                    (
                        environment_descriptor_set.clone(),
                        frame_descriptor_set.clone().offsets([offset]),
                    ),
                )
                .unwrap();

            unsafe { builder.draw_indexed(self.cube_index_buffer.len() as u32, 1, 0, 0, 0) }
                .unwrap();

            builder.end_render_pass(Default::default()).unwrap();
        }
    }

    fn generate_prefiltered_environment_map(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        environment_map: Arc<ImageView>,
        prefiltered_environment_map_image: Arc<Image>,
    ) {
        let proj = Mat4::perspective_rh_gl(90f32.to_radians(), 1.0, 0.01, 10.0);
        let view_projs = [
            proj * Mat4::look_at_rh(Vec3::ZERO, Vec3::X, Vec3::NEG_Y),
            proj * Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_X, Vec3::NEG_Y),
            proj * Mat4::look_at_rh(Vec3::ZERO, Vec3::Y, Vec3::Z),
            proj * Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Y, Vec3::NEG_Z),
            proj * Mat4::look_at_rh(Vec3::ZERO, Vec3::Z, Vec3::NEG_Y),
            proj * Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::NEG_Y),
        ];

        let environment_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.prefiltered_environment_map_generation_pipeline
                .layout()
                .set_layouts()[0]
                .clone(),
            [WriteDescriptorSet::image_view(1, environment_map.clone())],
            [],
        )
        .unwrap();
        
        let min_dynamic_align = self
            .device
            .physical_device()
            .properties()
            .min_uniform_buffer_offset_alignment
            .as_devicesize();

        let align =
            (size_of::<cubemap_vs::FrameUniforms>() as DeviceSize + min_dynamic_align - 1)
                & !(min_dynamic_align - 1);

        let uniform_buffer = self.uniform_buffer_allocator.allocate_slice(6 * align).unwrap();
        for (view_proj_index, view_proj) in view_projs.iter().enumerate() {
            *uniform_buffer
                .clone()
                .slice(view_proj_index as DeviceSize * align..)
                .cast_aligned()
                .index(0)
                .write()
                .unwrap() = cubemap_vs::FrameUniforms {
                    view_proj: view_proj.to_cols_array_2d(),
                };
        }

        let frame_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.prefiltered_environment_map_generation_pipeline
                .layout()
                .set_layouts()[1]
                .clone(),
            [WriteDescriptorSet::buffer_with_range(
                0,
                DescriptorBufferInfo {
                    buffer: uniform_buffer.clone(),
                    range: 0..size_of::<cubemap_vs::FrameUniforms>() as DeviceSize,
                },
            )],
            [],
        )
        .unwrap();

        for mip_level in 0..prefiltered_environment_map_image.mip_levels() {
            for view_proj_index in 0..view_projs.len() {
                let image_view_create_info = ImageViewCreateInfo {
                    view_type: ImageViewType::Dim2d,
                    subresource_range: ImageSubresourceRange {
                        array_layers: view_proj_index as u32..view_proj_index as u32 + 1,
                        mip_levels: mip_level..mip_level + 1,
                        ..prefiltered_environment_map_image.subresource_range()
                    },
                    ..ImageViewCreateInfo::from_image(&prefiltered_environment_map_image)
                };
                let image_view = ImageView::new(
                    prefiltered_environment_map_image.clone(),
                    image_view_create_info,
                )
                .unwrap();

                let extent = [
                    prefiltered_environment_map_image.extent()[0] / 2u32.pow(mip_level),
                    prefiltered_environment_map_image.extent()[1] / 2u32.pow(mip_level),
                ];
                let framebuffer = Framebuffer::new(
                    self.prefiltered_environment_map_generation_render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![image_view],
                        extent,
                        ..Default::default()
                    },
                )
                .unwrap();

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                            ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                        },
                        Default::default(),
                    )
                    .unwrap()
                    .set_viewport(
                        0,
                        [Viewport {
                            offset: [0.0, 0.0],
                            extent: [
                                framebuffer.extent()[0] as f32,
                                framebuffer.extent()[1] as f32,
                            ],
                            depth_range: 0.0..=1.0,
                        }]
                        .into_iter()
                        .collect(),
                    )
                    .unwrap()
                    .bind_pipeline_graphics(self.prefiltered_environment_map_generation_pipeline.clone())
                    .unwrap();

                builder
                    .bind_vertex_buffers(0, self.cube_vertex_buffer.clone())
                    .unwrap()
                    .bind_index_buffer(self.cube_index_buffer.clone())
                    .unwrap();

                let offset = (view_proj_index as DeviceSize * align) as u32;
                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        self.prefiltered_environment_map_generation_pipeline.layout().clone(),
                        0,
                        (
                            environment_descriptor_set.clone(),
                            frame_descriptor_set.clone().offsets([offset]),
                        ),
                    )
                    .unwrap();

                let roughness = mip_level as f32 / (prefiltered_environment_map_image.mip_levels() - 1) as f32;
                builder
                    .push_constants(
                        self.prefiltered_environment_map_generation_pipeline.layout().clone(),
                        0,
                        prefiltered_environment_map_generation_fs::RoughnessData {
                            roughness,
                        }
                    )
                    .unwrap();

                unsafe { builder.draw_indexed(self.cube_index_buffer.len() as u32, 1, 0, 0, 0) }
                    .unwrap();

                builder.end_render_pass(Default::default()).unwrap();
            }
        }
    }
    
    fn generate_environment_brdf_lut(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        environment_brdf_lut_image: Arc<Image>,
    ) {
        let image_view_create_info = ImageViewCreateInfo {
            view_type: ImageViewType::Dim2d,
            ..ImageViewCreateInfo::from_image(&environment_brdf_lut_image)
        };
        let image_view = ImageView::new(environment_brdf_lut_image, image_view_create_info).unwrap();

        let framebuffer = Framebuffer::new(
            self.environment_brdf_lut_generation_render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![image_view],
                ..Default::default()
            },
        )
        .unwrap();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                Default::default(),
            )
            .unwrap()
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [
                        framebuffer.extent()[0] as f32,
                        framebuffer.extent()[1] as f32,
                    ],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap()
            .bind_pipeline_graphics(self.environment_brdf_lut_generation_pipeline.clone())
            .unwrap();

        builder
            .bind_vertex_buffers(0, self.quad_vertex_buffer.clone())
            .unwrap()
            .bind_index_buffer(self.quad_index_buffer.clone())
            .unwrap();

        unsafe { builder.draw_indexed(self.quad_index_buffer.len() as u32, 1, 0, 0, 0) }
            .unwrap();

        builder.end_render_pass(Default::default()).unwrap();
    }

    // TODO: make this return a future of when it finishes rendering
    pub fn render_to_image(
        &self,
        environment_map: Arc<ImageView>,
        irradiance_map_image: Arc<Image>,
        prefiltered_environment_map_image: Arc<Image>,
        environment_brdf_lut_image: Arc<Image>,
    ) {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        self.generate_maps(&mut builder, environment_map.clone(), irradiance_map_image);
        self.generate_prefiltered_environment_map(&mut builder, environment_map.clone(), prefiltered_environment_map_image);
        self.generate_environment_brdf_lut(&mut builder, environment_brdf_lut_image);

        let command_buffer = builder.build().unwrap();

        command_buffer
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }
}

/// Shader without inverted x for rendering to another cubemap
mod cubemap_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "assets/shaders/cubemap.vs.glsl",
    }
}

mod irradiance_map_generation_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "assets/shaders/irradiance_map_convolution_fragment.glsl",
    }
}

mod prefiltered_environment_map_generation_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "assets/shaders/prefiltered_environment_map_generation.fs.glsl",
    }
}

mod environment_brdf_lut_generation {
    pub mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "assets/shaders/environment_brdf_lut_generation.vs.glsl",
        }
    }

    pub mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "assets/shaders/environment_brdf_lut_generation.fs.glsl",
        }
    }
}
