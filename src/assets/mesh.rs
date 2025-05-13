use std::{path::Path, sync::Arc};

use gltf::{mesh::util::ReadIndices, Gltf};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
};

#[derive(BufferContents, vulkano::pipeline::graphics::vertex_input::Vertex)]
#[repr(C)]
pub struct Vertex {
    #[format(R32G32B32_SFLOAT)]
    a_position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    a_normal: [f32; 3],
    #[format(R32G32_SFLOAT)]
    a_uv: [f32; 2],
}

pub struct Mesh {
    pub vertex_buffer: Subbuffer<[Vertex]>,
    pub index_buffer: Subbuffer<[u32]>,
}

impl Mesh {
    fn new(
        allocator: Arc<dyn MemoryAllocator>,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
    ) -> Self {
        let vertex_buffer = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        ).unwrap();

        let index_buffer = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            indices,
        ).unwrap();

        Mesh {
            vertex_buffer,
            index_buffer
        }
    }

    pub fn from_path(allocator: Arc<dyn MemoryAllocator>, path: &Path) -> Result<Self, String> {
        let gltf = Gltf::open(path)
            .map_err(|gltf_error| {
                format!(
                    "Error opening gltf file `{}`: {}",
                    path.to_str().unwrap(),
                    gltf_error.to_string(),
                )
            })?;
        let base_path = path.parent().unwrap_or_else(|| Path::new("./"));
        let buffers = gltf::import_buffers(&gltf.document, Some(base_path), gltf.blob)
            .map_err(|gltf_error| {
                format!(
                    "Error opening gltf buffers for file `{}`: {}",
                    path.to_str().unwrap(),
                    gltf_error.to_string(),
                )
            })?;

        let mesh = gltf.document.meshes().next()
            .ok_or(format!("{} has no meshes", path.to_str().unwrap()))?;

        let vertices = mesh.primitives()
            .flat_map(|primitive| {
                let positions_iter = primitive.reader(|buffer| Some(&buffers[buffer.index()])).read_positions().unwrap();
                let normals_iter = primitive.reader(|buffer| Some(&buffers[buffer.index()])).read_normals().unwrap();
                let uvs_iter = primitive.reader(|buffer| Some(&buffers[buffer.index()])).read_tex_coords(0).unwrap().into_f32();

                positions_iter.zip(normals_iter).zip(uvs_iter)
                    .map(|((a_position, a_normal), a_uv)| {
                        //println!("UV: ({}, {})", a_uv[0], a_uv[1]);
                        Vertex { a_position, a_normal, a_uv }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let first_primitive = mesh.primitives().next()
            .ok_or(format!("{} mesh has no primitives", path.to_str().unwrap()))?;

        let indices_reader = first_primitive
            .reader(|buffer| Some(&buffers[buffer.index()])).read_indices()
            .unwrap();

        let indices: Vec<u32> = match indices_reader {
            ReadIndices::U8(iter) => iter.map(|index| index.into()).collect(),
            ReadIndices::U16(iter) => iter.map(|index| index.into()).collect(),
            ReadIndices::U32(iter) => iter.collect(),
        };

        Ok(Self::new(allocator, vertices, indices))
    }
}

#[test]
fn test_import_gltf_mesh_from_path() -> Result<(), gltf::Error> {
    Ok(())
}
