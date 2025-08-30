use std::{
    sync::{Arc, RwLock},
    u32,
};

use ambient_occlusion::{
    ambient_occlusion_shaders, create_ambient_occlusion_framebuffer,
    create_ambient_occlusion_pipeline, create_ambient_occlusion_render_pass,
};
use flecs_ecs::prelude::*;
use glam::{Mat4, Vec2, Vec3};
use light_culling::{create_light_culling_pipeline, light_culling_shader};
use main_pass::{
    create_g_buffer_framebuffer, create_main_render_pass, create_mesh_pipeline,
    create_skybox_pipeline, mesh_shaders, skybox_shaders, SLOW_CHANGING_DESCRIPTOR_SET,
    SLOW_CHANGING_DESCRIPTOR_SET_MATERIALS_BUFFER_BINDING,
    SLOW_CHANGING_DESCRIPTOR_SET_TEXTURES_BINDING,
};
use post_process::{
    create_post_processing_pipeline, create_post_processing_render_pass, post_processing_shaders,
};
use shadow_maps::{
    create_shadow_map_framebuffer, create_shadow_map_pipeline, create_shadow_map_renderpass,
    shadow_mapping_shaders, SHADOW_MAP_CASCADE_COUNT,
};
use thiserror::Error;
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        Buffer, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        DrawIndexedIndirectCommand, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
        SubpassBeginInfo, SubpassEndInfo,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
    format::Format,
    image::{
        sampler::{
            BorderColor, Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, LOD_CLAMP_NONE,
        },
        view::ImageView,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{graphics::depth_stencil::CompareOp, ComputePipeline, GraphicsPipeline, Pipeline},
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::{EntryPoint, ShaderModule},
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
        self, DirectionalLight, DirectionalLightShadowMap, MaterialComponent, MeshComponent,
        Transform,
    },
};

mod ambient_occlusion;
mod light_culling;
mod main_pass;
mod post_process;
mod shadow_maps;

#[derive(Error, Debug)]
pub enum RendererError {
    #[error("{0}")]
    VulkanoVulkanError(#[from] Validated<VulkanError>),
    #[error("{0}")]
    VulkanoValidationError(#[from] Box<ValidationError>),
}

#[derive(Clone)]
pub struct RendererSettings {
    pub light_culling_max_lights_per_tile: u32,
    pub light_culling_tile_size: u32,
    pub light_culling_z_slices: u32,
    pub cascaded_shadow_map_level_size: u32,
    pub sample_count_per_level: [u32; SHADOW_MAP_CASCADE_COUNT as usize],
    pub shadow_bias: f32,
    pub shadow_slope_bias: f32,
    pub shadow_normal_bias: f32,
    pub penumbra_max_size: f32,
    pub shadow_map_cascade_split_lambda: f32,
    pub sample_count: u32,
    pub sample_radius: f32,
    pub ambient_occlusion_intensity: f32,
}

impl Default for RendererSettings {
    fn default() -> Self {
        Self {
            light_culling_max_lights_per_tile: 64,
            light_culling_tile_size: 32,
            light_culling_z_slices: 32,
            cascaded_shadow_map_level_size: 1536,
            sample_count_per_level: [10, 8, 2, 2],
            shadow_bias: 0.0005,
            shadow_slope_bias: 0.0005,
            shadow_normal_bias: 0.012,
            penumbra_max_size: 0.0015,
            shadow_map_cascade_split_lambda: 0.75,
            sample_count: 16,
            sample_radius: 0.5,
            ambient_occlusion_intensity: 1.0,
        }
    }
}

pub struct Renderer {
    settings: RendererSettings,

    device: Arc<Device>,

    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    uniform_buffer_allocator: SubbufferAllocator,
    host_writable_storage_buffer_allocator: SubbufferAllocator,
    in_device_storage_buffer_allocator: SubbufferAllocator,
    indirect_buffer_allocator: SubbufferAllocator,

    asset_database: Arc<RwLock<AssetDatabase>>,

    g_buffer: Arc<Framebuffer>,
    ambient_occlusion_buffer: Arc<Framebuffer>,

    render_pass: Arc<RenderPass>,
    mesh_vs: EntryPoint,
    mesh_fs: EntryPoint,
    mesh_pipeline: Arc<GraphicsPipeline>,
    skybox_vs: EntryPoint,
    skybox_fs: EntryPoint,
    skybox_pipeline: Arc<GraphicsPipeline>,

    full_screen_quad_vs: EntryPoint,
    post_processing_fs: EntryPoint,
    ambient_occlusion_fs: EntryPoint,
    post_processing_render_pass: Arc<RenderPass>,
    post_processing_pipeline: Arc<GraphicsPipeline>,
    ambient_occlusion_render_pass: Arc<RenderPass>,
    ambient_occlusion_pipeline: Arc<GraphicsPipeline>,

    shadow_mapping_render_pass: Arc<RenderPass>,
    shadow_mapping_vs: EntryPoint,
    shadow_mapping_fs: EntryPoint,
    shadow_mapping_pipeline: Arc<GraphicsPipeline>,

    shadow_map_framebuffer: Arc<Framebuffer>,

    /// For tiled rendering
    light_culling_pipeline: Arc<ComputePipeline>,
    light_culling_cs_module: Arc<ShaderModule>,
    old_light_culling_max_lights_per_tile: u32,

    cube_vertex_buffer: Subbuffer<[Vertex]>,
    cube_index_buffer: Subbuffer<[u32]>,

    full_screen_vertex_buffer: Subbuffer<[Vertex]>,
    full_screen_index_buffer: Subbuffer<[u32]>,

    asset_change_listener: Arc<RwLock<RendererAssetChangeListener>>,

    nearest_sampler_float: Arc<Sampler>,
    nearest_sampler_any: Arc<Sampler>,
    sampler: Arc<Sampler>,
    shadow_map_sampler: Arc<Sampler>,

    materials_storage_buffer: Subbuffer<[mesh_shaders::fs::Material]>,
    slow_changing_descriptor_set: Arc<DescriptorSet>,

    environment_cubemap_query: Query<&'static components::EnvironmentCubemap>,
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
        settings: RendererSettings,
        device: Arc<Device>,
        queue: Arc<Queue>,
        asset_database: Arc<RwLock<AssetDatabase>>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        world: World,
        format: Format,
    ) -> Result<Self, RendererError> {
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

        let light_culling_cs_module = light_culling_shader::cs::load(device.clone())?;

        let light_culling_pipeline = create_light_culling_pipeline(
            &device,
            &light_culling_cs_module,
            settings.light_culling_max_lights_per_tile,
        )?;

        let shadow_mapping_render_pass = create_shadow_map_renderpass(&device)?;

        let shadow_mapping_vs = shadow_mapping_shaders::vs::load(device.clone())?
            .entry_point("main")
            .unwrap();
        let shadow_mapping_fs = shadow_mapping_shaders::fs::load(device.clone())?
            .entry_point("main")
            .unwrap();

        let shadow_map_framebuffer = create_shadow_map_framebuffer(
            &memory_allocator,
            &shadow_mapping_render_pass,
            settings.cascaded_shadow_map_level_size,
        )?;

        let shadow_mapping_pipeline = create_shadow_map_pipeline(
            Vec2::from(shadow_map_framebuffer.extent().map(|x| x as f32)),
            &shadow_mapping_render_pass,
            &memory_allocator,
            &shadow_mapping_vs,
            &shadow_mapping_fs,
        )?;

        let render_pass = create_main_render_pass(&device)?;

        let mesh_vs = mesh_shaders::vs::load(device.clone())?
            .entry_point("main")
            .unwrap();
        let mesh_fs = mesh_shaders::fs::load(device.clone())?
            .entry_point("main")
            .unwrap();

        let skybox_vs = skybox_shaders::vs::load(device.clone())?
            .entry_point("main")
            .unwrap();
        let skybox_fs = skybox_shaders::fs::load(device.clone())?
            .entry_point("main")
            .unwrap();

        let full_screen_quad_vs = full_screen_quad::vs::load(device.clone())?
            .entry_point("main")
            .unwrap();
        let post_processing_fs = post_processing_shaders::fs::load(device.clone())?
            .entry_point("main")
            .unwrap();

        let ambient_occlusion_fs = ambient_occlusion_shaders::fs::load(device.clone())?
            .entry_point("main")
            .unwrap();

        let nearest_sampler_float = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                lod: 0.0..=LOD_CLAMP_NONE,
                address_mode: [SamplerAddressMode::ClampToBorder; 3],
                border_color: BorderColor::FloatOpaqueWhite,
                ..Default::default()
            },
        )?;

        let nearest_sampler_any = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                lod: 0.0..=LOD_CLAMP_NONE,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )?;

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                lod: 0.0..=LOD_CLAMP_NONE,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )?;

        let shadow_map_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                lod: 0.0..=LOD_CLAMP_NONE,
                address_mode: [SamplerAddressMode::Repeat; 3],
                compare: Some(CompareOp::Less),
                ..Default::default()
            },
        )?;

        let extent = Vec2::new(1280.0, 720.0);

        let post_processing_render_pass = create_post_processing_render_pass(&device, format)?;
        let post_processing_pipeline = create_post_processing_pipeline(
            extent,
            Subpass::from(post_processing_render_pass.clone(), 0).unwrap(),
            &device,
            &full_screen_quad_vs,
            &post_processing_fs,
            &nearest_sampler_any,
        )?;

        let g_buffer = create_g_buffer_framebuffer(
            &render_pass,
            &memory_allocator,
            [extent.x as u32, extent.y as u32, 1],
        )?;

        let mesh_pipeline = create_mesh_pipeline(
            extent,
            Subpass::from(render_pass.clone(), 0).unwrap(),
            &device,
            &mesh_vs,
            &mesh_fs,
            &sampler,
            &shadow_map_sampler,
        )?;

        let skybox_pipeline = create_skybox_pipeline(
            extent,
            Subpass::from(render_pass.clone(), 1).unwrap(),
            &device,
            &skybox_vs,
            &skybox_fs,
            &sampler,
        )?;

        let ambient_occlusion_render_pass = create_ambient_occlusion_render_pass(&device)?;
        let ambient_occlusion_pipeline = create_ambient_occlusion_pipeline(
            extent,
            Subpass::from(ambient_occlusion_render_pass.clone(), 0).unwrap(),
            &device,
            &full_screen_quad_vs,
            &ambient_occlusion_fs,
            &nearest_sampler_float,
            &nearest_sampler_any,
        )?;

        let ambient_occlusion_buffer = create_ambient_occlusion_framebuffer(
            &ambient_occlusion_render_pass,
            &memory_allocator,
            [extent.x as u32, extent.y as u32, 1],
        )?;

        let (cube_vertex_buffer, cube_index_buffer) = load_mesh_from_buffers_into_new_buffers(
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            &queue,
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

        let (full_screen_vertex_buffer, full_screen_index_buffer) =
            load_mesh_from_buffers_into_new_buffers(
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                &queue,
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

        let environment_cubemap_query = world
            .query::<&components::EnvironmentCubemap>()
            .term_at(0)
            .singleton()
            .build();

        let meshes_with_materials_query =
            world.new_query::<(&MeshComponent, &MaterialComponent, &Transform)>();

        assert!(size_of::<mesh_shaders::fs::Material>() % 16 == 0);

        let materials_storage_buffer = Buffer::new_slice(
            memory_allocator.clone(),
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
        let environment_descriptor_set = DescriptorSet::new_variable(
            descriptor_set_allocator.clone(),
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

        let renderer = Self {
            settings: settings.clone(),

            device,

            memory_allocator,
            descriptor_set_allocator,
            uniform_buffer_allocator,
            host_writable_storage_buffer_allocator,
            in_device_storage_buffer_allocator,
            indirect_buffer_allocator,
            asset_database,

            g_buffer,
            ambient_occlusion_buffer,

            render_pass,
            mesh_vs,
            mesh_fs,
            mesh_pipeline,
            skybox_vs,
            skybox_fs,
            skybox_pipeline,

            full_screen_quad_vs,
            post_processing_fs,
            ambient_occlusion_fs,
            post_processing_render_pass,
            post_processing_pipeline,
            ambient_occlusion_render_pass,
            ambient_occlusion_pipeline,

            shadow_mapping_render_pass,
            shadow_mapping_pipeline,
            shadow_mapping_vs,
            shadow_mapping_fs,
            shadow_map_framebuffer,

            light_culling_pipeline,
            light_culling_cs_module,
            old_light_culling_max_lights_per_tile: settings.light_culling_max_lights_per_tile,

            cube_vertex_buffer,
            cube_index_buffer,

            full_screen_vertex_buffer,
            full_screen_index_buffer,

            nearest_sampler_float,
            nearest_sampler_any,
            sampler,
            shadow_map_sampler,
            materials_storage_buffer,
            slow_changing_descriptor_set: environment_descriptor_set,
            asset_change_listener,
            environment_cubemap_query,
            meshes_with_materials_query,
            world,
        };

        Ok(renderer)
    }

    pub fn settings_mut(&mut self) -> &mut RendererSettings {
        &mut self.settings
    }

    fn add_materials_and_textures_to_descriptor_set(&mut self) -> Result<(), RendererError> {
        let asset_change_listener_read = self.asset_change_listener.read().unwrap();
        let textures_changed = asset_change_listener_read.textures_changed;
        let materials_changed = asset_change_listener_read.materials_changed;
        drop(asset_change_listener_read);

        if !textures_changed && !materials_changed {
            return Ok(());
        }

        if materials_changed {
            let asset_database_read = self.asset_database.read().unwrap();

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

            self.materials_storage_buffer = self
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

        let asset_database_read = self.asset_database.read().unwrap();
        self.slow_changing_descriptor_set = DescriptorSet::new_variable(
            self.descriptor_set_allocator.clone(),
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
        self.g_buffer =
            create_g_buffer_framebuffer(&self.render_pass, &self.memory_allocator, extent)?;

        self.ambient_occlusion_buffer = create_ambient_occlusion_framebuffer(
            &self.ambient_occlusion_render_pass,
            &self.memory_allocator,
            extent,
        )?;

        let extent = Vec2::new(
            image_views[0].image().extent()[0] as f32,
            image_views[0].image().extent()[1] as f32,
        );

        self.mesh_pipeline = create_mesh_pipeline(
            extent,
            Subpass::from(self.render_pass.clone(), 0).unwrap(),
            &self.device,
            &self.mesh_vs,
            &self.mesh_fs,
            &self.sampler,
            &self.shadow_map_sampler,
        )?;

        self.skybox_pipeline = create_skybox_pipeline(
            extent,
            Subpass::from(self.render_pass.clone(), 1).unwrap(),
            &self.device,
            &self.skybox_vs,
            &self.skybox_fs,
            &self.sampler,
        )?;

        self.ambient_occlusion_pipeline = create_ambient_occlusion_pipeline(
            extent,
            Subpass::from(self.ambient_occlusion_render_pass.clone(), 0).unwrap(),
            &self.device,
            &self.full_screen_quad_vs,
            &self.ambient_occlusion_fs,
            &self.nearest_sampler_float,
            &self.nearest_sampler_any,
        )?;

        self.post_processing_pipeline = create_post_processing_pipeline(
            extent,
            Subpass::from(self.post_processing_render_pass.clone(), 0).unwrap(),
            &self.device,
            &self.full_screen_quad_vs,
            &self.post_processing_fs,
            &self.nearest_sampler_any,
        )?;

        Ok(self.create_framebuffers(&image_views)?)
    }

    pub fn apply_changed_settings(&mut self) -> Result<(), RendererError> {
        if self.settings.light_culling_max_lights_per_tile
            != self.old_light_culling_max_lights_per_tile
        {
            self.light_culling_pipeline = create_light_culling_pipeline(
                &self.device,
                &self.light_culling_cs_module,
                self.settings.light_culling_max_lights_per_tile,
            )?;
            self.old_light_culling_max_lights_per_tile =
                self.settings.light_culling_max_lights_per_tile;
        }

        if self.settings.cascaded_shadow_map_level_size != self.shadow_map_framebuffer.extent()[0] {
            self.shadow_map_framebuffer = create_shadow_map_framebuffer(
                &self.memory_allocator,
                &self.shadow_mapping_render_pass,
                self.settings.cascaded_shadow_map_level_size,
            )?;
            self.shadow_mapping_pipeline = create_shadow_map_pipeline(
                Vec2::from(self.shadow_map_framebuffer.extent().map(|x| x as f32)),
                &self.shadow_mapping_render_pass,
                &self.memory_allocator,
                &self.shadow_mapping_vs,
                &self.shadow_mapping_fs,
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

        self.add_materials_and_textures_to_descriptor_set()?;
        let point_lights_buffer = self.create_point_lights_storage_buffer();
        let directional_lights_buffer = self.create_directional_lights_storage_buffer();

        self.apply_changed_settings()?;

        let shadow_map_matrices_and_splits =
            self.compute_shadow_map_matrices_and_splits(&camera, aspect_ratio);

        let (visible_light_indices_storage_buffer, lights_from_tile_storage_image_view) =
            self.cull_lights(builder, &camera, aspect_ratio, &point_lights_buffer)?;

        let entities_data_and_indirect_commands =
            self.get_entities_data_and_indirect_draw_commands();

        if let Some((entities_data, indirect_commands)) =
            entities_data_and_indirect_commands.as_ref()
        {
            if let Some((_depth_splits, light_space_matrices)) = shadow_map_matrices_and_splits {
                self.render_shadow_maps(
                    builder,
                    &entities_data,
                    &indirect_commands,
                    &light_space_matrices,
                )?;
            }
        }

        // main pass
        builder.begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0u32; 2].into()), Some(1f32.into())],
                ..RenderPassBeginInfo::framebuffer(self.g_buffer.clone())
            },
            SubpassBeginInfo::default(),
        )?;

        if let Some((entities_data, indirect_commands)) =
            entities_data_and_indirect_commands.as_ref()
        {
            self.draw_meshes(
                builder,
                &camera,
                aspect_ratio,
                &point_lights_buffer,
                &directional_lights_buffer,
                &visible_light_indices_storage_buffer,
                &lights_from_tile_storage_image_view,
                &entities_data,
                &indirect_commands,
                &shadow_map_matrices_and_splits,
            )?;
        }

        builder.next_subpass(SubpassEndInfo::default(), SubpassBeginInfo::default())?;

        self.draw_skybox(builder, &camera, aspect_ratio)?;

        builder.end_render_pass(Default::default())?;

        self.ambient_occlusion_pass(builder, &camera, aspect_ratio)?;

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
            .indirect_buffer_allocator
            .allocate_slice(indirect_commands.len() as DeviceSize)
            .unwrap();
        indirect_buffer
            .write()
            .unwrap()
            .copy_from_slice(&indirect_commands);

        let entity_data_storage_buffer = self
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
