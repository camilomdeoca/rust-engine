use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::CommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferToImageInfo, PrimaryCommandBufferAbstract,
    },
    device::Queue,
    format::Format,
    image::{view::{ImageView, ImageViewCreateInfo, ImageViewType}, Image, ImageCreateFlags, ImageCreateInfo, ImageType, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryAllocatePreference, MemoryAllocator, MemoryTypeFilter},
    sync::GpuFuture,
};

pub fn load_texture_from_buffer(
    memory_allocator: &Arc<impl MemoryAllocator + ?Sized>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    queue: Arc<Queue>,
    format: Format,
    extent: [u32; 2],
    pixel_data: &[u8],
) -> Result<Arc<ImageView>, String> {
    load_texture_from_buffer_impl(
        memory_allocator,
        command_buffer_allocator,
        queue,
        format,
        extent,
        pixel_data.iter().cloned(),
        1,
        ImageCreateFlags::empty(),
        ImageViewType::Dim2d,
    )
}

pub fn load_cubemap_from_buffer(
    memory_allocator: &Arc<impl MemoryAllocator + ?Sized>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    queue: Arc<Queue>,
    format: Format,
    extent: [u32; 2],
    pixel_data: &[u8],
) -> Result<Arc<ImageView>, String> {
    load_texture_from_buffer_impl(
        memory_allocator,
        command_buffer_allocator,
        queue,
        format,
        extent,
        pixel_data.iter().cloned(),
        6,
        ImageCreateFlags::CUBE_COMPATIBLE,
        ImageViewType::Cube,
    )
}

fn load_texture_from_buffer_impl<I>(
    memory_allocator: &Arc<impl MemoryAllocator + ?Sized>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    queue: Arc<Queue>,
    format: Format,
    extent: [u32; 2],
    pixel_data: I,
    array_layers: u32,
    flags: ImageCreateFlags,
    view_type: ImageViewType,
) -> Result<Arc<ImageView>, String>
where
    I: IntoIterator<Item = u8>,
    I::IntoIter: ExactSizeIterator,
{
    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.clone().queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let staging_buffer = Buffer::from_iter(
        &memory_allocator,
        &BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        &AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        pixel_data,
    )
    .unwrap();

    let image = Image::new(
        &memory_allocator,
        &ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format,
            extent: [extent[0], extent[1], 1],
            array_layers,
            flags,
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        &AllocationCreateInfo {
            allocate_preference: MemoryAllocatePreference::AlwaysAllocate,
            ..Default::default()
        },
    )
    .expect(
        format!(
            "Error creating image of size: {}x{} with format {:?}",
            extent[0], extent[1], format,
        )
        .as_str(),
    );

    builder
        .copy_buffer_to_image(CopyBufferToImageInfo::new(
            staging_buffer,
            image.clone(),
        ))
        .unwrap();

    let image_view_create_info = ImageViewCreateInfo {
        view_type,
        ..ImageViewCreateInfo::from_image(&image)
    };
    let image_view = ImageView::new(&image, &image_view_create_info).unwrap();

    let command_buffer = builder.build().unwrap();

    command_buffer
        .execute(queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    Ok(image_view)
}
