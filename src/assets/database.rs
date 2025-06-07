use std::{collections::HashMap, path::Path, sync::Arc};

use gltf::{mesh::util::ReadIndices, Gltf};
use image::EncodableLayout;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer}, command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfo, PrimaryCommandBufferAbstract,
    }, device::Queue, format::Format, image::view::ImageView, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, sync::GpuFuture, DeviceSize
};

use super::{loaders::texture_loader::{load_cubemap_from_buffer, load_texture_from_buffer}, vertex::Vertex};

pub struct AssetDatabase {
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    memory_allocator: Arc<StandardMemoryAllocator>,

    meshes: HashMap<String, (Subbuffer<[Vertex]>, Subbuffer<[u32]>)>,
    textures: HashMap<String, Arc<ImageView>>,
    cubemaps: HashMap<String, Arc<ImageView>>,
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
            meshes: HashMap::new(),
            textures: HashMap::new(),
            cubemaps: HashMap::new(),
        }
    }

    pub fn load_mesh_from_buffers(
        &mut self,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
        name: impl AsRef<str>,
    ) -> Result<(Subbuffer<[Vertex]>, Subbuffer<[u32]>), String> {
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

        self.meshes.insert(
            name.as_ref().to_string(),
            (vertex_buffer.clone(), index_buffer.clone()),
        );

        Ok((vertex_buffer, index_buffer))
    }

    pub fn load_mesh(
        &mut self,
        path: impl AsRef<Path>,
        name: impl AsRef<str>,
    ) -> Result<(Subbuffer<[Vertex]>, Subbuffer<[u32]>), String> {
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

        self.load_mesh_from_buffers(vertices, indices, name)
    }

    pub fn load_texture(
        &mut self,
        path: impl AsRef<Path>,
        name: impl AsRef<str>,
    ) -> Result<Arc<ImageView>, String> {
        let format = Format::R8G8B8A8_UNORM;
        let image = image::open(path).unwrap().into_rgba8();
        let texture = load_texture_from_buffer(
            self.memory_allocator.clone(),
            self.command_buffer_allocator.clone(),
            self.queue.clone(),
            format,
            image.dimensions().into(),
            &image,
        )?;

        self.textures.insert(
            name.as_ref().to_string(),
            texture.clone(),
        );

        Ok(texture)
    }

    pub fn load_cubemap(
        &mut self,
        paths: [impl AsRef<Path>; 6],
        name: impl AsRef<str>,
    ) -> Result<Arc<ImageView>, String> {
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

        let cubemap;
        if format == Format::R16G16B16A16_UNORM {
            let mut data = Vec::new();
            let mut dimensions = (0, 0);
            for path in paths {
                let face = image::open(path).unwrap().into_rgba16();
                dimensions = face.dimensions();
                data.extend_from_slice(face.as_bytes());
            }
            
            cubemap = load_cubemap_from_buffer(
                self.memory_allocator.clone(),
                self.command_buffer_allocator.clone(),
                self.queue.clone(),
                format,
                dimensions.into(),
                &data,
            )?;
        } else if format == Format::R32G32B32A32_SFLOAT {
            let mut data = Vec::new();
            let mut dimensions = (0, 0);
            for path in paths {
                let face = image::open(path).unwrap().into_rgba32f();
                dimensions = face.dimensions();
                data.extend_from_slice(face.as_bytes());
            }
            
            cubemap = load_cubemap_from_buffer(
                self.memory_allocator.clone(),
                self.command_buffer_allocator.clone(),
                self.queue.clone(),
                format,
                dimensions.into(),
                &data,
            )?;
        } else {
            let mut data = Vec::new();
            let mut dimensions = (0, 0);
            for path in paths {
                let face = image::open(path).unwrap().into_rgba8();
                dimensions = face.dimensions();
                data.extend_from_slice(face.as_bytes());
            }
            
            cubemap = load_cubemap_from_buffer(
                self.memory_allocator.clone(),
                self.command_buffer_allocator.clone(),
                self.queue.clone(),
                format,
                dimensions.into(),
                &data,
            )?;
        }
        self.cubemaps.insert(
            name.as_ref().to_string(),
            cubemap.clone(),
        );

        Ok(cubemap)
    }

    pub fn get_mesh(&self, name: &str) -> Option<(Subbuffer<[Vertex]>, Subbuffer<[u32]>)> {
        self.meshes.get(name).cloned()
    }

    pub fn get_texture(&self, name: &str) -> Option<Arc<ImageView>> {
        self.textures.get(name).cloned()
    }

    pub fn get_cubemap(&self, name: &str) -> Option<Arc<ImageView>> {
        self.cubemaps.get(name).cloned()
    }
}
