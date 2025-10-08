use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo, PrimaryAutoCommandBuffer},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    DeviceSize,
};

pub fn load_mesh_from_buffers<VI, II, Vertex>(
    memory_allocator: Arc<dyn MemoryAllocator>,
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    vertices: VI,
    indices: II,
    vertex_buffer: Subbuffer<[Vertex]>,
    index_buffer: Subbuffer<[u32]>,
) -> Result<(), String>
where
    VI: IntoIterator<Item = Vertex>,
    VI::IntoIter: ExactSizeIterator,
    II: IntoIterator<Item = u32>,
    II::IntoIter: ExactSizeIterator,
    Vertex: vulkano::pipeline::graphics::vertex_input::Vertex,
{
    let staging_vertex_buffer = Buffer::from_iter(
        memory_allocator.clone(),
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
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        indices,
    )
    .unwrap();

    assert!(staging_vertex_buffer.len() <= vertex_buffer.len());
    assert!(staging_index_buffer.len() <= index_buffer.len());

    builder
        .copy_buffer(CopyBufferInfo::buffers(
            staging_vertex_buffer,
            vertex_buffer,
        ))
        .unwrap()
        .copy_buffer(CopyBufferInfo::buffers(staging_index_buffer, index_buffer))
        .unwrap();

    Ok(())
}

pub fn load_mesh_from_buffers_into_new_buffers<VI, II, Vertex>(
    memory_allocator: Arc<dyn MemoryAllocator>,
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    vertices: VI,
    indices: II,
) -> Result<(Subbuffer<[Vertex]>, Subbuffer<[u32]>), String>
where
    VI: IntoIterator<Item = Vertex>,
    VI::IntoIter: ExactSizeIterator,
    II: IntoIterator<Item = u32>,
    II::IntoIter: ExactSizeIterator,
    Vertex: vulkano::pipeline::graphics::vertex_input::Vertex,
{
    let vertices = vertices.into_iter();
    let indices = indices.into_iter();

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
        vertices.len() as DeviceSize,
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
        indices.len() as DeviceSize,
    )
    .unwrap();

    load_mesh_from_buffers(
        memory_allocator,
        builder,
        vertices,
        indices,
        vertex_buffer.clone(),
        index_buffer.clone(),
    )?;

    Ok((vertex_buffer, index_buffer))
}
