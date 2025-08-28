use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::CommandBufferAllocator, AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, CopyBufferToImageInfo, ImageBlit, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract
    },
    device::Queue,
    format::Format,
    image::{
        sampler::Filter, view::{ImageView, ImageViewCreateInfo, ImageViewType}, Image, ImageCreateFlags, ImageCreateInfo, ImageLayout, ImageSubresourceLayers, ImageType, ImageUsage
    },
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    sync::GpuFuture,
};

pub fn load_texture_from_buffer(
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    queue: Arc<Queue>,
    format: Format,
    extent: [u32; 2],
    pixel_data: &[u8],
) -> Result<Arc<ImageView>, String> {
    let mip_levels = extent.iter().max().unwrap().ilog2();

    load_texture_from_buffer_impl(
        memory_allocator,
        command_buffer_allocator,
        queue,
        pixel_data.iter().cloned(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format,
            extent: [extent[0], extent[1], 1],
            mip_levels,
            usage: ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        ImageViewType::Dim2d,
    )
}

pub fn load_3d_texture_from_buffer(
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    queue: Arc<Queue>,
    format: Format,
    extent: [u32; 3],
    pixel_data: &[u8],
) -> Result<Arc<ImageView>, String> {
    load_texture_from_buffer_impl(
        memory_allocator,
        command_buffer_allocator,
        queue,
        pixel_data.iter().cloned(),
        ImageCreateInfo {
            image_type: ImageType::Dim3d,
            format,
            extent,
            usage: ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        ImageViewType::Dim3d,
    )
}

pub fn load_cubemap_from_buffer(
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    queue: Arc<Queue>,
    format: Format,
    extent: [u32; 2],
    pixel_data: &[u8],
) -> Result<Arc<ImageView>, String> {
    let mip_levels = extent.iter().max().unwrap().ilog2();

    load_texture_from_buffer_impl(
        memory_allocator,
        command_buffer_allocator,
        queue,
        pixel_data.iter().cloned(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format,
            extent: [extent[0], extent[1], 1],
            array_layers: 6,
            flags: ImageCreateFlags::CUBE_COMPATIBLE,
            mip_levels,
            usage: ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        ImageViewType::Cube,
    )
}

fn generate_mipmaps(
    image: &Arc<Image>,
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>
) {
    for i in 1..image.mip_levels() {
        builder
            .blit_image(BlitImageInfo {
                src_image: image.clone(),
                src_image_layout: ImageLayout::TransferSrcOptimal,
                dst_image: image.clone(),
                dst_image_layout: ImageLayout::TransferDstOptimal,
                regions: [ImageBlit {
                    src_subresource: ImageSubresourceLayers {
                        mip_level: i - 1,
                        ..image.subresource_layers()
                    },
                    src_offsets: [[0; 3], image.extent().map(|e| if e > 1 { e >> (i - 1) } else { e })],
                    dst_subresource: ImageSubresourceLayers {
                        mip_level: i,
                        ..image.subresource_layers()
                    },
                    dst_offsets: [[0; 3], image.extent().map(|e| if e > 1 { e >> i } else { e })],
                    ..Default::default()
                }].into(),
                filter: Filter::Linear,
                ..BlitImageInfo::images(image.clone(), image.clone())
            })
            .unwrap();
    }
}

fn load_texture_from_buffer_impl<I>(
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    queue: Arc<Queue>,
    pixel_data: I,
    create_info: ImageCreateInfo,
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
        pixel_data,
    )
    .unwrap();

    let image = Image::new(
        memory_allocator.clone(),
        create_info,
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .unwrap();

    builder
        .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            staging_buffer,
            image.clone(),
        ))
        .unwrap();

    generate_mipmaps(&image, &mut builder);

    let image_view_create_info = ImageViewCreateInfo {
        view_type,
        ..ImageViewCreateInfo::from_image(&image)
    };
    let image_view = ImageView::new(image, image_view_create_info).unwrap();

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
