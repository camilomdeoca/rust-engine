use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::CommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfo, PrimaryCommandBufferAbstract,
    },
    device::Queue,
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    sync::GpuFuture, DeviceSize,
};

pub fn load_mesh_from_buffers<VI, II, Vertex>(
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    queue: &Arc<Queue>,
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

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .copy_buffer(CopyBufferInfo::buffers(
            staging_vertex_buffer,
            vertex_buffer,
        ))
        .unwrap()
        .copy_buffer(CopyBufferInfo::buffers(
            staging_index_buffer,
            index_buffer,
        ))
        .unwrap();

    let command_buffer = builder.build().unwrap();

    command_buffer
        .execute(queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    Ok(())
}

pub fn load_mesh_from_buffers_into_new_buffers<VI, II, Vertex>(
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    queue: &Arc<Queue>,
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
        command_buffer_allocator,
        queue,
        vertices,
        indices,
        vertex_buffer.clone(),
        index_buffer.clone(),
    )?;

    Ok((vertex_buffer, index_buffer))
}
