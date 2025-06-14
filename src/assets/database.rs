use std::{path::Path, sync::Arc};

use bimap::BiHashMap;
use glam::{UVec2, Vec3, Vec4};
use gltf::{mesh::util::ReadIndices, Gltf};
use image::EncodableLayout;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer}, command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfo, PrimaryCommandBufferAbstract,
    }, device::Queue, format::Format, image::view::ImageView, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, sync::GpuFuture, DeviceSize
};

use super::{loaders::texture_loader::{load_cubemap_from_buffer, load_texture_from_buffer}, vertex::Vertex};

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct MeshId(u32);

pub struct Mesh {
    pub vertex_buffer: Subbuffer<[Vertex]>,
    pub index_buffer: Subbuffer<[u32]>,
}

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct TextureId(u32);

pub struct Texture {
    pub texture: Arc<ImageView>,
}

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct CubemapId(u32);

pub struct Cubemap {
    pub cubemap: Arc<ImageView>,
}

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct MaterialId(u32);

pub struct Material {
    pub color_factor: Vec4,
    pub diffuse: Option<TextureId>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub metallic_roughness: Option<TextureId>,
    pub ambient_oclussion: Option<TextureId>,
    pub emissive_factor: Vec3,
    pub emissive: Option<TextureId>,
    pub normal: Option<TextureId>,
}

pub struct AssetDatabase {
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    memory_allocator: Arc<StandardMemoryAllocator>,

    mesh_names: BiHashMap<String, MeshId>,
    meshes: Vec<Mesh>,

    texture_names: BiHashMap<String, TextureId>,
    textures: Vec<Texture>,
    
    cubemap_names: BiHashMap<String, CubemapId>,
    cubemaps: Vec<Cubemap>,
    
    material_names: BiHashMap<String, MaterialId>,
    materials: Vec<Material>,
}

impl AssetDatabase {
    pub fn new(queue: Arc<Queue>, memory_allocator: Arc<StandardMemoryAllocator>) -> Self {
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            queue.device().clone(),
            Default::default(),
        ));

        Self {
            queue,
            command_buffer_allocator,
            memory_allocator,
            mesh_names: BiHashMap::new(),
            meshes: Vec::new(),
            texture_names: BiHashMap::new(),
            textures: Vec::new(),
            cubemap_names: BiHashMap::new(),
            cubemaps: Vec::new(),
            material_names: BiHashMap::new(),
            materials: Vec::new(),
        }
    }

    pub fn add_mesh_from_buffers(
        &mut self,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
    ) -> Result<MeshId, String> {
        let vertex_count = vertices.len();
        let index_count = indices.len();

        let staging_vertex_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();

        let staging_index_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            indices.iter().cloned(),
        )
        .unwrap();

        let vertex_buffer = Buffer::new_slice(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            vertex_count as DeviceSize,
        )
        .unwrap();

        let index_buffer = Buffer::new_slice(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            index_count as DeviceSize,
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                staging_vertex_buffer,
                vertex_buffer.clone(),
            ))
            .unwrap()
            .copy_buffer(CopyBufferInfo::buffers(
                staging_index_buffer,
                index_buffer.clone(),
            ))
            .unwrap();

        let command_buffer = builder.build().unwrap();

        command_buffer
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let mesh_id = MeshId(self.meshes.len() as u32);
        self.meshes.push(Mesh {
            vertex_buffer,
            index_buffer,
        });

        Ok(mesh_id)
    }
    
    pub fn add_named_mesh_from_buffers(
        &mut self,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
        name: impl AsRef<str>,
    ) -> Result<MeshId, String> {
        if let Some(mesh_id) = self.mesh_names.get_by_left(name.as_ref()) {
            Ok(mesh_id.clone())
        } else {
            let result = self.add_mesh_from_buffers(vertices, indices);
            if result.is_ok() {
                self.mesh_names.insert(name.as_ref().to_string(), result.clone().unwrap());
            }
            result
        }
    }

    pub fn add_mesh_from_path(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<MeshId, String> {
        let path = path.as_ref();
        let gltf = Gltf::open(path).map_err(|gltf_error| {
            format!(
                "Error opening gltf file `{}`: {}",
                path.to_str().unwrap(),
                gltf_error.to_string(),
            )
        })?;
        let base_path = path.parent().unwrap_or_else(|| Path::new("./"));
        let gltf_buffers = gltf::import_buffers(&gltf.document, Some(base_path), gltf.blob)
            .map_err(|gltf_error| {
                format!(
                    "Error opening gltf buffers for file `{}`: {}",
                    path.to_str().unwrap(),
                    gltf_error.to_string(),
                )
            })?;

        let mesh = gltf
            .document
            .meshes()
            .next()
            .ok_or(format!("{} has no meshes", path.to_str().unwrap()))?;

        let vertices = mesh
            .primitives()
            .flat_map(|primitive| {
                let positions_iter = primitive
                    .reader(|buffer| Some(&gltf_buffers[buffer.index()]))
                    .read_positions()
                    .unwrap();
                let normals_iter = primitive
                    .reader(|buffer| Some(&gltf_buffers[buffer.index()]))
                    .read_normals()
                    .unwrap();
                let tangents_iter = primitive
                    .reader(|buffer| Some(&gltf_buffers[buffer.index()]))
                    .read_tangents()
                    .unwrap();
                let uvs_iter = primitive
                    .reader(|buffer| Some(&gltf_buffers[buffer.index()]))
                    .read_tex_coords(0)
                    .unwrap()
                    .into_f32();

                positions_iter.zip(normals_iter).zip(tangents_iter).zip(uvs_iter).map(
                    |(((a_position, a_normal), a_tangent), a_uv)| Vertex {
                        a_position,
                        a_normal,
                        a_tangent: [a_tangent[0], a_tangent[1], a_tangent[2]],
                        a_uv,
                    },
                )
            })
            .collect();

        let first_primitive = mesh
            .primitives()
            .next()
            .ok_or(format!("{} mesh has no primitives", path.to_str().unwrap()))?;

        let indices_reader = first_primitive
            .reader(|buffer| Some(&gltf_buffers[buffer.index()]))
            .read_indices()
            .unwrap();

        let indices = match indices_reader {
            ReadIndices::U8(iter) => iter.map(|index| index.into()).collect(),
            ReadIndices::U16(iter) => iter.map(|index| index.into()).collect(),
            ReadIndices::U32(iter) => iter.collect(),
        };

        self.add_mesh_from_buffers(vertices, indices)
    }
    
    pub fn add_named_mesh_from_path(
        &mut self,
        path: impl AsRef<Path>,
        name: impl AsRef<str>,
    ) -> Result<MeshId, String> {
        if let Some(mesh_id) = self.mesh_names.get_by_left(name.as_ref()) {
            Ok(mesh_id.clone())
        } else {
            let result = self.add_mesh_from_path(path);
            if result.is_ok() {
                self.mesh_names.insert(name.as_ref().to_string(), result.clone().unwrap());
            }
            result
        }
    }

    pub fn add_texture_from_buffer(
        &mut self,
        format: Format,
        dimensions: UVec2,
        pixel_data: &[u8],
    ) -> Result<TextureId, String> {
        let texture = load_texture_from_buffer(
            self.memory_allocator.clone(),
            self.command_buffer_allocator.clone(),
            self.queue.clone(),
            format,
            dimensions.into(),
            pixel_data,
        )?;

        let texture_id = TextureId(self.textures.len() as u32);
        self.textures.push(Texture { texture });

        Ok(texture_id)
    }
    
    pub fn add_named_texture_from_buffer(
        &mut self,
        format: Format,
        dimensions: UVec2,
        pixel_data: &[u8],
        name: impl AsRef<str>,
    ) -> Result<TextureId, String> {
        if let Some(texture_id) = self.texture_names.get_by_left(name.as_ref()) {
            Ok(texture_id.clone())
        } else {
            let result = self.add_texture_from_buffer(format, dimensions, pixel_data);
            if result.is_ok() {
                self.texture_names.insert(name.as_ref().to_string(), result.clone().unwrap());
            }
            result
        }
    }

    pub fn add_texture_from_path(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<TextureId, String> {
        let format = Format::R8G8B8A8_UNORM;
        let image = image::open(path).unwrap().into_rgba8();
        self.add_texture_from_buffer(format, image.dimensions().into(), &image)
    }
    
    pub fn add_named_texture_from_path(
        &mut self,
        path: impl AsRef<Path>,
        name: impl AsRef<str>,
    ) -> Result<TextureId, String> {
        if let Some(texture_id) = self.texture_names.get_by_left(name.as_ref()) {
            Ok(texture_id.clone())
        } else {
            let result = self.add_texture_from_path(path);
            if result.is_ok() {
                self.texture_names.insert(name.as_ref().to_string(), result.clone().unwrap());
            }
            result
        }
    }

    pub fn add_cubemap_from_raw(
        &mut self,
        cubemap: Arc<ImageView>,
    ) -> Result<CubemapId, String> {
        let cubemap_id = CubemapId(self.cubemaps.len() as u32);
        self.cubemaps.push(Cubemap { cubemap });

        Ok(cubemap_id)
    }

    pub fn add_cubemap_from_buffer(
        &mut self,
        format: Format,
        dimensions: UVec2,
        pixel_data: &[u8],
    ) -> Result<CubemapId, String> {
        let cubemap = load_cubemap_from_buffer(
            self.memory_allocator.clone(),
            self.command_buffer_allocator.clone(),
            self.queue.clone(),
            format,
            dimensions.into(),
            pixel_data,
        )?;
        
        self.add_cubemap_from_raw(cubemap)
    }

    pub fn add_cubemap_from_path(
        &mut self,
        paths: [impl AsRef<Path>; 6],
    ) -> Result<CubemapId, String> {
        let format = match image::open(paths[0].as_ref()).unwrap().color() {
            image::ColorType::Rgb8 => Format::R8G8B8A8_SRGB,
            image::ColorType::Rgba8 => Format::R8G8B8A8_SRGB,
            image::ColorType::Rgb16 => Format::R16G16B16A16_UNORM,
            image::ColorType::Rgba16 => Format::R16G16B16A16_UNORM,
            image::ColorType::Rgb32F => Format::R32G32B32A32_SFLOAT,
            image::ColorType::Rgba32F => Format::R32G32B32A32_SFLOAT,
            color_type => panic!("Unsupported color type {color_type:?}"),
        };
        // let format = Format::R8G8B8A8_SRGB;

        let mut data = Vec::new();
        let mut dimensions = UVec2::ZERO;
        if format == Format::R16G16B16A16_UNORM {
            for path in paths {
                let face = image::open(path).unwrap().into_rgba16();
                dimensions = face.dimensions().into();
                data.extend_from_slice(face.as_bytes());
            }
        } else if format == Format::R32G32B32A32_SFLOAT {
            for path in paths {
                let face = image::open(path).unwrap().into_rgba32f();
                dimensions = face.dimensions().into();
                data.extend_from_slice(face.as_bytes());
            }
        } else {
            for path in paths {
                let face = image::open(path).unwrap().into_rgba8();
                dimensions = face.dimensions().into();
                data.extend_from_slice(face.as_bytes());
            }
            
        }

        self.add_cubemap_from_buffer(format, dimensions, &data)
    }

    pub fn add_material(
        &mut self,
        color_factor: Vec4,
        diffuse: Option<TextureId>,
        metallic_factor: f32,
        roughness_factor: f32,
        metallic_roughness: Option<TextureId>,
        ambient_oclussion: Option<TextureId>,
        emissive_factor: Vec3,
        emissive: Option<TextureId>,
        normal: Option<TextureId>,
    ) -> Result<MaterialId, String> {
        let material_id = MaterialId(self.materials.len() as u32);
        self.materials.push(Material {
            color_factor,
            diffuse,
            metallic_factor,
            roughness_factor,
            metallic_roughness,
            ambient_oclussion,
            emissive_factor,
            emissive,
            normal,
        });
        Ok(material_id)
    }

    pub fn get_mesh(&self, mesh_id: MeshId) -> Option<&Mesh> {
        self.meshes.get(mesh_id.0 as usize)
    }

    pub fn get_texture(&self, texture_id: TextureId) -> Option<&Texture> {
        self.textures.get(texture_id.0 as usize)
    }

    pub fn get_cubemap(&self, cubemap_id: CubemapId) -> Option<&Cubemap> {
        self.cubemaps.get(cubemap_id.0 as usize)
    }
    
    pub fn get_material(&self, material_id: MaterialId) -> Option<&Material> {
        self.materials.get(material_id.0 as usize)
    }
}
