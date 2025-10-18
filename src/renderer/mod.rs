use std::{
    sync::{Arc, RwLock},
    u32,
};

use ambient_occlusion::AmbientOcclusionPass;
use cascaded_shadow_maps_pass::{CascadedShadowMapsPass, CascadedShadowMapsPassError};
use flecs_ecs::prelude::*;
use glam::{uvec2, Mat4, UVec2, Vec2, Vec3};
use light_culling::LightCullingPass;
use main_pass::{mesh_shaders, MeshesPass};
use post_process::{
    create_post_processing_pipeline, create_post_processing_render_pass, post_processing_shaders,
};
use thiserror::Error;
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        BufferUsage, Subbuffer,
    },
    command_buffer::{
        AutoCommandBufferBuilder, DrawIndexedIndirectCommand, PrimaryAutoCommandBuffer,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{Device, DeviceOwned},
    format::Format,
    image::{
        sampler::{
            Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, LOD_CLAMP_NONE,
        },
        view::ImageView,
    },
    memory::allocator::{MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::GraphicsPipeline,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::EntryPoint,
    DeviceSize, Validated, ValidationError, VulkanError,
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
        DirectionalLight, DirectionalLightShadowMap, MaterialComponent, MeshComponent,
        Transform,
    },
    settings::Settings,
};

mod ambient_occlusion;
mod cascaded_shadow_maps_pass;
mod light_culling;
mod main_pass;
mod post_process;

#[derive(Error, Debug)]
pub enum RendererError {
    #[error("{0}")]
    VulkanoVulkanError(#[from] Validated<VulkanError>),
    #[error("{0}")]
    VulkanoValidationError(#[from] Box<ValidationError>),
    #[error("{0}")]
    CascadedShadowMapsPassError(#[from] CascadedShadowMapsPassError),
}

struct RendererContext {
    device: Arc<Device>,

    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    uniform_buffer_allocator: SubbufferAllocator,
    host_writable_storage_buffer_allocator: SubbufferAllocator,
    in_device_storage_buffer_allocator: SubbufferAllocator,
    indirect_buffer_allocator: SubbufferAllocator,
}

// trait RendererPass {
//     type Input;
//     type Output;
//
//     fn run(&mut self, context: &RendererContext, input: &Self::Input) -> Result<Self::Output, RendererError>;
// }

pub struct Renderer {
    settings: Settings,

    context: RendererContext,
    // device: Arc<Device>,
    //
    // memory_allocator: Arc<StandardMemoryAllocator>,
    // descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    // uniform_buffer_allocator: SubbufferAllocator,
    // host_writable_storage_buffer_allocator: SubbufferAllocator,
    // in_device_storage_buffer_allocator: SubbufferAllocator,
    // indirect_buffer_allocator: SubbufferAllocator,
    asset_database: Arc<RwLock<AssetDatabase>>,

    g_buffer: Arc<Framebuffer>,
    ambient_occlusion_buffer: Arc<Framebuffer>,

    cascaded_shadow_maps_pass: CascadedShadowMapsPass,
    light_culling_pass: LightCullingPass,
    meshes_pass: MeshesPass,
    ambient_occlusion_pass: AmbientOcclusionPass,

    full_screen_quad_vs: EntryPoint,
    post_processing_fs: EntryPoint,
    post_processing_render_pass: Arc<RenderPass>,
    post_processing_pipeline: Arc<GraphicsPipeline>,

    shadow_map_framebuffer: Arc<Framebuffer>,

    /// For tiled rendering
    old_light_culling_max_lights_per_tile: u32,

    full_screen_vertex_buffer: Subbuffer<[Vertex]>,
    full_screen_index_buffer: Subbuffer<[u32]>,

    nearest_sampler_any: Arc<Sampler>,

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
        settings: Settings,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        asset_database: Arc<RwLock<AssetDatabase>>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        world: World,
        format: Format,
    ) -> Result<Self, RendererError> {
        let context = {
            let device = memory_allocator.device().clone();
            let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
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

            RendererContext {
                device,
                memory_allocator: memory_allocator.clone(),
                descriptor_set_allocator,
                uniform_buffer_allocator,
                host_writable_storage_buffer_allocator,
                in_device_storage_buffer_allocator,
                indirect_buffer_allocator,
            }
        };

        let full_screen_quad_vs = full_screen_quad::vs::load(context.device.clone())?
            .entry_point("main")
            .unwrap();
        let post_processing_fs = post_processing_shaders::fs::load(context.device.clone())?
            .entry_point("main")
            .unwrap();

        let nearest_sampler_any = Sampler::new(
            context.device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                lod: 0.0..=LOD_CLAMP_NONE,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )?;

        let extent = Vec2::new(1280.0, 720.0);

        let post_processing_render_pass =
            create_post_processing_render_pass(&context.device, format)?;
        let post_processing_pipeline = create_post_processing_pipeline(
            extent,
            Subpass::from(post_processing_render_pass.clone(), 0).unwrap(),
            &context.device,
            &full_screen_quad_vs,
            &post_processing_fs,
            &nearest_sampler_any,
        )?;

        // let g_buffer = create_g_buffer_framebuffer(
        //     &render_pass,
        //     &memory_allocator,
        //     [extent.x as u32, extent.y as u32, 1],
        // )?;

        // let ambient_occlusion_render_pass = create_ambient_occlusion_render_pass(&context.device)?;
        // let ambient_occlusion_pipeline = create_ambient_occlusion_pipeline(
        //     extent,
        //     Subpass::from(ambient_occlusion_render_pass.clone(), 0).unwrap(),
        //     &context.device,
        //     &full_screen_quad_vs,
        //     &ambient_occlusion_fs,
        //     &nearest_sampler_float,
        //     &nearest_sampler_any,
        // )?;
        //
        // let ambient_occlusion_buffer = create_ambient_occlusion_framebuffer(
        //     &ambient_occlusion_render_pass,
        //     &memory_allocator,
        //     [extent.x as u32, extent.y as u32, 1],
        // )?;

        let (full_screen_vertex_buffer, full_screen_index_buffer) =
            load_mesh_from_buffers_into_new_buffers(
                memory_allocator.clone(),
                builder,
                vec![
                    Vertex {
                        a_position: [-1.0, -1.0, 0.5],
                        a_normal: [0.0; 3],
                        a_tangent: [0.0; 3],
                        a_uv: [0.0; 2],
                    },
                    Vertex {
                        a_position: [1.0, -1.0, 0.5],
                        a_normal: [0.0; 3],
                        a_tangent: [0.0; 3],
                        a_uv: [0.0; 2],
                    },
                    Vertex {
                        a_position: [-1.0, 1.0, 0.5],
                        a_normal: [0.0; 3],
                        a_tangent: [0.0; 3],
                        a_uv: [0.0; 2],
                    },
                    Vertex {
                        a_position: [1.0, 1.0, 0.5],
                        a_normal: [0.0; 3],
                        a_tangent: [0.0; 3],
                        a_uv: [0.0; 2],
                    },
                ],
                vec![2, 1, 0, 3, 1, 2],
            )
            .unwrap();

        let meshes_with_materials_query =
            world.new_query::<(&MeshComponent, &MaterialComponent, &Transform)>();

        assert!(size_of::<mesh_shaders::fs::Material>() % 16 == 0);

        let cascaded_shadow_maps_pass = CascadedShadowMapsPass::new(&context, &settings)?;
        let light_culling_pass = LightCullingPass::new(&context, &settings)?;
        let meshes_pass = MeshesPass::new(builder, &context, extent, &asset_database, &world)?;
        let ambient_occlusion_pass = AmbientOcclusionPass::new(&context, extent.as_uvec2())?;

        let shadow_map_framebuffer = cascaded_shadow_maps_pass.create_framebuffer(
            &context,
            settings.renderer.shadow_mapping.cascade_level_size,
        )?;

        let g_buffer = meshes_pass.create_g_buffer_framebuffer(&context, extent.as_uvec2())?;

        let ambient_occlusion_buffer = ambient_occlusion_pass.create_framebuffer(
            &context,
            extent.as_uvec2(),
        )?;

        let asset_change_listener = Arc::new(RwLock::new(RendererAssetChangeListener {
            textures_changed: false,
            materials_changed: false,
        }));

        asset_database
            .write()
            .unwrap()
            .add_asset_database_change_observer(asset_change_listener.clone());

        let renderer = Self {
            settings: settings.clone(),

            context,

            asset_database,

            g_buffer,
            ambient_occlusion_buffer,

            light_culling_pass,
            cascaded_shadow_maps_pass,
            meshes_pass,
            ambient_occlusion_pass,

            full_screen_quad_vs,
            post_processing_fs,
            post_processing_render_pass,
            post_processing_pipeline,

            shadow_map_framebuffer,

            old_light_culling_max_lights_per_tile: settings
                .renderer
                .light_culling
                .max_lights_per_tile,

            full_screen_vertex_buffer,
            full_screen_index_buffer,

            nearest_sampler_any,
            meshes_with_materials_query,
            world,
        };

        Ok(renderer)
    }

    pub fn settings_mut(&mut self) -> &mut Settings {
        &mut self.settings
    }

    fn create_directional_lights_storage_buffer(
        &mut self,
    ) -> Option<Subbuffer<[mesh_shaders::fs::DirectionalLight]>> {
        let mut directional_lights = vec![];

        let query = self
            .world
            .query::<(&DirectionalLight, &Transform)>()
            .with::<DirectionalLightShadowMap>()
            .optional()
            .build();

        query.run(|mut it| {
            while it.next() {
                let directional_light = it.field::<&DirectionalLight>(0).unwrap();
                let transform = it.field::<&Transform>(1).unwrap();
                let has_shadow_maps = it.is_set(2); // DirectionalLightShadowMap

                for i in it.iter() {
                    directional_lights.push(mesh_shaders::fs::DirectionalLight {
                        direction: (transform[i].rotation * Vec3::NEG_Z).to_array().into(),
                        color: directional_light[i].color.to_array().into(),
                        has_shadow_maps: has_shadow_maps.into(),
                    });
                }
            }
        });

        if directional_lights.is_empty() {
            None
        } else {
            let directional_lights_storage_buffer = self
                .context
                .host_writable_storage_buffer_allocator
                .allocate_slice(directional_lights.len() as DeviceSize)
                .unwrap();

            directional_lights_storage_buffer
                .write()
                .unwrap()
                .copy_from_slice(&directional_lights);

            Some(directional_lights_storage_buffer)
        }
    }

    fn create_framebuffers(
        &self,
        image_views: &[Arc<ImageView>],
    ) -> Result<Vec<Arc<Framebuffer>>, RendererError> {
        let framebuffers = image_views
            .iter()
            .map(|image_view| {
                Framebuffer::new(
                    self.post_processing_render_pass.clone(),
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

    pub fn resize_and_create_framebuffers(
        &mut self,
        image_views: &[Arc<ImageView>],
    ) -> Result<Vec<Arc<Framebuffer>>, RendererError> {
        let extent = image_views[0].image().extent();
        self.g_buffer = self
            .meshes_pass
            .resize_and_create_g_buffer(&self.context, UVec2::new(extent[0], extent[1]))?;

        self.ambient_occlusion_pass = AmbientOcclusionPass::new(
            &self.context,
            uvec2(extent[0], extent[1]),
        )?;

        self.ambient_occlusion_buffer = self.ambient_occlusion_pass.create_framebuffer(
            &self.context,
            uvec2(extent[0], extent[1]),
        )?;

        let extent = Vec2::new(
            image_views[0].image().extent()[0] as f32,
            image_views[0].image().extent()[1] as f32,
        );

        self.post_processing_pipeline = create_post_processing_pipeline(
            extent,
            Subpass::from(self.post_processing_render_pass.clone(), 0).unwrap(),
            &self.context.device,
            &self.full_screen_quad_vs,
            &self.post_processing_fs,
            &self.nearest_sampler_any,
        )?;

        Ok(self.create_framebuffers(&image_views)?)
    }

    pub fn apply_changed_settings(&mut self) -> Result<(), RendererError> {
        if self.settings.renderer.light_culling.max_lights_per_tile
            != self.old_light_culling_max_lights_per_tile
        {
            self.light_culling_pass = LightCullingPass::new(&self.context, &self.settings)?;
            self.old_light_culling_max_lights_per_tile =
                self.settings.renderer.light_culling.max_lights_per_tile;
        }

        if self.settings.renderer.shadow_mapping.cascade_level_size
            != self.shadow_map_framebuffer.extent()[0]
        {
            self.cascaded_shadow_maps_pass =
                CascadedShadowMapsPass::new(&self.context, &self.settings)?;
            self.shadow_map_framebuffer = self.cascaded_shadow_maps_pass.create_framebuffer(
                &self.context,
                self.settings.renderer.shadow_mapping.cascade_level_size,
            )?;
        }

        Ok(())
    }

    /// Returns true if the swapchain needs to be recreated
    pub fn draw(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        framebuffer: &Arc<Framebuffer>,
        camera: &Camera,
    ) -> Result<(), RendererError> {
        let aspect_ratio = framebuffer.extent()[0] as f32 / framebuffer.extent()[1] as f32;

        let point_lights_buffer = self.create_point_lights_storage_buffer();
        let directional_lights_buffer = self.create_directional_lights_storage_buffer();

        self.apply_changed_settings()?;

        // let shadow_map_matrices_and_splits =
        //     self.compute_shadow_map_matrices_and_splits(&camera, aspect_ratio);

        let g_buffer_extent = uvec2(
            self.g_buffer.attachments()[0].image().extent()[0],
            self.g_buffer.attachments()[0].image().extent()[1],
        );

        let (visible_light_indices_storage_buffer, lights_from_tile_storage_image_view) =
            self.light_culling_pass.execute(
                builder,
                &self.context,
                &camera,
                aspect_ratio,
                &point_lights_buffer,
                g_buffer_extent,
                &self.settings,
            )?;

        let entities_data_and_indirect_commands =
            self.get_entities_data_and_indirect_draw_commands();

        let asset_database_read = self.asset_database.read().unwrap();
        let vertex_buffer = asset_database_read.vertex_buffer().clone();
        let index_buffer = asset_database_read.index_buffer().clone();
        drop(asset_database_read);

        if let Some((entities_data, indirect_commands)) =
            entities_data_and_indirect_commands.as_ref()
        {
            let (shadow_maps_texture, shadow_map_matrices_and_splits) = self.cascaded_shadow_maps_pass.execute(
                builder,
                &self.context,
                &self.shadow_map_framebuffer,
                &entities_data,
                &indirect_commands,
                &vertex_buffer,
                &index_buffer,
                &camera,
                aspect_ratio,
                &self.world,
                &self.settings,
            )?;
            // main pass
            self.meshes_pass.run(
                builder,
                &self.context,
                camera,
                aspect_ratio,
                &point_lights_buffer,
                &directional_lights_buffer,
                &visible_light_indices_storage_buffer,
                &lights_from_tile_storage_image_view,
                &shadow_maps_texture,
                &shadow_map_matrices_and_splits,
                &self.asset_database,
                &self.settings,
                &self.g_buffer,
                &self.world,
            )?;
        }

        self.ambient_occlusion_pass.execute(
            builder,
            &self.context,
            &self.ambient_occlusion_buffer,
            &self.g_buffer.attachments()[0],
            &self.g_buffer.attachments()[1],
            &self.full_screen_vertex_buffer,
            &self.full_screen_index_buffer,
            &camera,
            aspect_ratio,
            &self.settings,
        )?;

        self.post_processing_pass(builder, framebuffer)?;

        Ok(())
    }

    fn get_entities_data_and_indirect_draw_commands(
        &self,
    ) -> Option<(
        Subbuffer<[mesh_shaders::vs::EntityData]>,
        Subbuffer<[DrawIndexedIndirectCommand]>,
    )> {
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
            return None;
        }

        let indirect_buffer = self
            .context
            .indirect_buffer_allocator
            .allocate_slice(indirect_commands.len() as DeviceSize)
            .unwrap();
        indirect_buffer
            .write()
            .unwrap()
            .copy_from_slice(&indirect_commands);

        let entity_data_storage_buffer = self
            .context
            .host_writable_storage_buffer_allocator
            .allocate_slice(entities_data.len() as DeviceSize)
            .unwrap();
        entity_data_storage_buffer
            .write()
            .unwrap()
            .copy_from_slice(&entities_data);
        Some((entity_data_storage_buffer, indirect_buffer))
    }
}

mod full_screen_quad {
    pub mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "assets/shaders/full_screen_quad.vs.glsl",
        }
    }
}
