use std::{
    path::Path,
    sync::{Arc, RwLock},
};

use bimap::BiHashMap;
use glam::{UVec2, Vec3, Vec4};
use image::EncodableLayout;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CopyBufferInfo, PrimaryAutoCommandBuffer},
    device::Queue,
    format::Format,
    image::view::ImageView,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    DeviceSize,
};

use super::{
    loaders::texture_loader::{load_cubemap_from_buffer, load_texture_from_buffer},
    vertex::Vertex,
};

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct MeshId(pub u32);

pub struct Mesh {
    pub index_count: u32,
    pub first_index: u32,
    pub vertex_count: u32, // not needed for rendering but for knowing where to write next mesh in
    // vertex_buffer
    pub vertex_offset: u32,
}

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct TextureId(pub u32);

pub struct Texture {
    pub texture: Arc<ImageView>,
}

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct CubemapId(pub u32);

pub struct Cubemap {
    pub cubemap: Arc<ImageView>,
}

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct MaterialId(pub u32);

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

pub trait AssetDatabaseChangeObserver: Send + Sync {
    fn on_mesh_add(&mut self, mesh_id: MeshId, mesh: &Mesh) {}
    fn on_texture_add(&mut self, texture_id: TextureId, texture: &Texture) {}
    fn on_cubemap_add(&mut self, cubemap_id: CubemapId, cubemap: &Cubemap) {}
    fn on_material_add(&mut self, material_id: MaterialId, material: &Material) {}
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

    vertex_buffer: Subbuffer<[Vertex]>,
    index_buffer: Subbuffer<[u32]>,

    add_mesh_to_main_buffers_queue: Vec<AddMeshToMainBufferTask>,

    asset_database_change_observers: Vec<Arc<RwLock<dyn AssetDatabaseChangeObserver>>>,
}

struct AddMeshToMainBufferTask {
    /// Id of the mesh we are going to add
    mesh_id: MeshId,

    /// Staging buffers that will be copied to the main buffers
    staging_vertex_buffer: Subbuffer<[Vertex]>,
    staging_index_buffer: Subbuffer<[u32]>,
}

impl AssetDatabase {
    pub fn new(
        queue: Arc<Queue>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        vertex_buffer_len: DeviceSize,
        index_buffer_len: DeviceSize,
    ) -> Self {
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            queue.device().clone(),
            Default::default(),
        ));

        let vertex_buffer = Buffer::new_slice(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            vertex_buffer_len,
        )
        .unwrap();

        let index_buffer = Buffer::new_slice(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            index_buffer_len,
        )
        .unwrap();

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
            add_mesh_to_main_buffers_queue: Vec::new(),
            asset_database_change_observers: Vec::new(),
            vertex_buffer,
            index_buffer,
        }
    }

    pub fn add_asset_database_change_observer(
        &mut self,
        observer: Arc<RwLock<dyn AssetDatabaseChangeObserver>>,
    ) {
        self.asset_database_change_observers.push(observer);
    }

    pub fn vertex_buffer(&self) -> Subbuffer<[Vertex]> {
        self.vertex_buffer.clone()
    }

    pub fn index_buffer(&self) -> Subbuffer<[u32]> {
        self.index_buffer.clone()
    }

    pub fn meshes(&self) -> &Vec<Mesh> {
        &self.meshes
    }

    pub fn textures(&self) -> &Vec<Texture> {
        &self.textures
    }

    pub fn materials(&self) -> &Vec<Material> {
        &self.materials
    }

    /// To call this you have to ensure the vertex and index buffers arent being used
    pub fn add_newly_loaded_meshes_to_main_buffers(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>
    ) {
        let mut vertex_buffer_offset: DeviceSize = self.meshes.iter().fold(0, |acc, mesh| {
            acc.max((mesh.vertex_offset + mesh.vertex_count) as DeviceSize)
        });

        let mut index_buffer_offset: DeviceSize = self.meshes.iter().fold(0, |acc, mesh| {
            acc.max((mesh.first_index + mesh.index_count) as DeviceSize)
        });

        let old_add_mesh_to_main_buffers_queue =
            std::mem::take(&mut self.add_mesh_to_main_buffers_queue);
        for task in old_add_mesh_to_main_buffers_queue {
            let AddMeshToMainBufferTask {
                mesh_id,
                staging_vertex_buffer,
                staging_index_buffer,
            } = task;

            assert!(staging_vertex_buffer.len() <= self.vertex_buffer.len() - vertex_buffer_offset);
            assert!(staging_index_buffer.len() <= self.index_buffer.len() - vertex_buffer_offset);

            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    staging_vertex_buffer.clone(),
                    self.vertex_buffer.clone().slice(vertex_buffer_offset..),
                ))
                .unwrap()
                .copy_buffer(CopyBufferInfo::buffers(
                    staging_index_buffer.clone(),
                    self.index_buffer.clone().slice(index_buffer_offset..),
                ))
                .unwrap();

            self.meshes[mesh_id.0 as usize] = Mesh {
                index_count: staging_index_buffer.len() as u32,
                first_index: index_buffer_offset as u32,
                vertex_count: staging_vertex_buffer.len() as u32,
                vertex_offset: vertex_buffer_offset as u32,
            };

            vertex_buffer_offset += staging_vertex_buffer.len();
            index_buffer_offset += staging_index_buffer.len();

            for observer in &self.asset_database_change_observers {
                observer
                    .write()
                    .unwrap()
                    .on_mesh_add(mesh_id.clone(), self.meshes.last().unwrap());
            }
        }
    }

    pub fn add_mesh_from_buffers(
        &mut self,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
    ) -> Result<MeshId, String> {
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
            vertices.iter().cloned(),
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

        let mesh_id = MeshId(self.meshes.len() as u32);
        self.meshes.push(Mesh {
            index_count: 0,
            first_index: 0,
            vertex_count: 0,
            vertex_offset: 0,
        });

        self.add_mesh_to_main_buffers_queue.push(AddMeshToMainBufferTask {
            mesh_id: mesh_id.clone(),
            staging_vertex_buffer,
            staging_index_buffer,
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
                self.mesh_names
                    .insert(name.as_ref().to_string(), result.clone().unwrap());
            }
            result
        }
    }

    pub fn add_texture_from_raw(&mut self, texture: Arc<ImageView>) -> Result<TextureId, String> {
        let texture_id = TextureId(self.textures.len() as u32);
        self.textures.push(Texture { texture });

        for observer in &self.asset_database_change_observers {
            observer
                .write()
                .unwrap()
                .on_texture_add(texture_id.clone(), self.textures.last().unwrap());
        }

        Ok(texture_id)
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

        self.add_texture_from_raw(texture)
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
                self.texture_names
                    .insert(name.as_ref().to_string(), result.clone().unwrap());
            }
            result
        }
    }

    pub fn add_texture_from_path(&mut self, path: impl AsRef<Path>) -> Result<TextureId, String> {
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
                self.texture_names
                    .insert(name.as_ref().to_string(), result.clone().unwrap());
            }
            result
        }
    }

    pub fn add_cubemap_from_raw(&mut self, cubemap: Arc<ImageView>) -> Result<CubemapId, String> {
        let cubemap_id = CubemapId(self.cubemaps.len() as u32);
        self.cubemaps.push(Cubemap { cubemap });

        for observer in &self.asset_database_change_observers {
            observer
                .write()
                .unwrap()
                .on_cubemap_add(cubemap_id.clone(), self.cubemaps.last().unwrap());
        }

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

        for observer in &self.asset_database_change_observers {
            observer
                .write()
                .unwrap()
                .on_material_add(material_id.clone(), self.materials.last().unwrap());
        }

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
