use std::sync::Arc;

use glam::{Mat4, UVec3, Vec3};
use vulkano::{
    buffer::Subbuffer,
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    device::Device,
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage},
    memory::allocator::AllocationCreateInfo,
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    shader::ShaderModule,
    DeviceSize, Validated, VulkanError,
};

use crate::{
    camera::Camera,
    ecs::components::{PointLight, Transform},
};

use super::{main_pass::mesh_shaders, Renderer, RendererError};

pub fn create_light_culling_pipeline(
    device: &Arc<Device>,
    light_culling_cs_module: &Arc<ShaderModule>,
    max_lights_per_tile: u32,
) -> Result<Arc<ComputePipeline>, Validated<VulkanError>> {
    let light_culling_cs = light_culling_cs_module
        .specialize([(0, max_lights_per_tile.into())].into_iter().collect())?
        .entry_point("main")
        .unwrap();

    let stage = PipelineShaderStageCreateInfo::new(light_culling_cs.clone());

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )?;

    ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
}

impl Renderer {
    pub fn create_point_lights_storage_buffer(
        &mut self,
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
            let point_lights_storage_buffer = self
                .host_writable_storage_buffer_allocator
                .allocate_slice(point_lights.len() as DeviceSize)
                .unwrap();

            point_lights_storage_buffer
                .write()
                .unwrap()
                .copy_from_slice(&point_lights);

            Some(point_lights_storage_buffer)
        }
    }

    pub fn cull_lights(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        camera: &Camera,
        aspect_ratio: f32,
        point_lights_storage_buffer: &Option<Subbuffer<[mesh_shaders::fs::PointLight]>>,
    ) -> Result<(Subbuffer<[u32]>, Arc<ImageView>), RendererError> {
        let next_ligth_index_global_storage_buffer = self
            .in_device_storage_buffer_allocator
            .allocate_slice::<light_culling_shader::cs::NextLigthIndexGlobal>(4) // allocating 1 freezes
            .unwrap();

        // Initialize with value 0
        assert_eq!(
            size_of::<light_culling_shader::cs::NextLigthIndexGlobal>(),
            size_of::<u32>()
        );
        builder.fill_buffer(
            next_ligth_index_global_storage_buffer
                .clone()
                .reinterpret::<[u32]>(),
            0,
        )?;

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
            width.div_ceil(self.settings.light_culling_tile_size),
            height.div_ceil(self.settings.light_culling_tile_size),
            self.settings.light_culling_z_slices,
        );

        let visible_light_indices_storage_buffer = self
            .in_device_storage_buffer_allocator
            .allocate_slice::<u32>(
                (num_tiles.x
                    * num_tiles.y
                    * num_tiles.z
                    * self.settings.light_culling_max_lights_per_tile)
                    as DeviceSize,
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
            ImageView::new_default(lights_from_tile_storage_image)?;

        let frame_uniform_buffer = {
            let view = Mat4::look_to_rh(
                camera.position,
                (camera.rotation * Vec3::NEG_Z).normalize(),
                (camera.rotation * Vec3::Y).normalize(),
            );
            let proj = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0))
                * Mat4::perspective_rh(camera.fov, aspect_ratio, 0.01, 100.0);

            let num_lights = match point_lights_storage_buffer {
                Some(point_lights_storage_buffer) => point_lights_storage_buffer.len() as _,
                None => 0,
            };

            let uniform_data = light_culling_shader::cs::FrameUniforms {
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
                        .map(|buf| buf.clone())
                        .unwrap_or_else(|| {
                            self.in_device_storage_buffer_allocator
                                .allocate_slice(1)
                                .unwrap()
                        }),
                ),
                WriteDescriptorSet::buffer(2, next_ligth_index_global_storage_buffer.clone()),
                WriteDescriptorSet::buffer(3, visible_light_indices_storage_buffer.clone()),
                WriteDescriptorSet::image_view(4, lights_from_tile_storage_image_view.clone()),
            ],
            [],
        )?;

        builder
            .bind_pipeline_compute(self.light_culling_pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.light_culling_pipeline.layout().clone(),
                0,
                set,
            )?;

        unsafe { builder.dispatch(num_tiles.to_array()) }?;

        Ok((
            visible_light_indices_storage_buffer,
            lights_from_tile_storage_image_view,
        ))
    }
}

pub mod light_culling_shader {
    pub mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "assets/shaders/light_culling.cs.glsl",
        }
    }
}
