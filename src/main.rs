// Welcome to the triangle example!
//
// This is the only example that is entirely detailed. All the other examples avoid code
// duplication by using helper functions.
//
// This example assumes that you are already more or less familiar with graphics programming and
// that you want to learn Vulkan. This means that for example it won't go into details about what a
// vertex or a shader is.

mod assets;
mod camera;
mod ecs;
mod image_based_lighting_maps_generator;
mod profile;
mod renderer;
mod ui;
mod input;

use assets::{
    database::AssetDatabase,
    loaders::gltf_scene_loader::{count_vertices_and_indices_in_gltf_scene, load_gltf_scene},
};
use camera::Camera;
use ecs::components::{EnvironmentCubemap, MaterialComponent, MeshComponent, SceneEntity, Transform};
use egui_winit_vulkano::{Gui, GuiConfig};
use flecs_ecs::prelude::*;
use glam::{EulerRot, Quat, Vec3};
use image_based_lighting_maps_generator::ImageBasedLightingMapsGenerator;
use log::{info, warn};
use profile::ProfileTimer;
use renderer::Renderer;
use ui::{logger, UserInterface};
use std::{
    collections::HashMap,
    error::Error,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags,
    },
    format::Format,
    image::{
        view::{ImageView, ImageViewCreateInfo, ImageViewType},
        Image, ImageCreateFlags, ImageCreateInfo, ImageType, ImageUsage,
    },
    instance::{
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger,
            DebugUtilsMessengerCallback, DebugUtilsMessengerCallbackData,
            DebugUtilsMessengerCreateInfo,
        },
        Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions,
    },
    memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator},
    swapchain::{
        acquire_next_image, PresentFuture, PresentMode, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo
    },
    sync::{self, future::FenceSignalFuture, GpuFuture},
    DeviceSize, Validated, VulkanError, VulkanLibrary,
};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, RawKeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    raw_window_handle::HasDisplayHandle,
    window::{CursorGrabMode, Window, WindowId},
};

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

struct RenderingContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    swapchain_image_views: Vec<Arc<ImageView>>,
    recreate_swapchain: bool,
    fences: Vec<Option<FenceSignalFuture<PresentFuture<Box<dyn GpuFuture>>>>>,
    previous_fence_index: u32,
    renderer: Arc<RwLock<Renderer>>,
    start_frame_instant: Instant,
    user_interface: UserInterface,
    frametime: Duration,
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,

    world: World,
    entities_to_add_queue: Arc<RwLock<Vec<(String, Transform, MeshComponent, MaterialComponent)>>>,
    asset_database: Arc<RwLock<AssetDatabase>>,
    _debug_callback: Arc<DebugUtilsMessenger>,

    rendering_context: Option<RenderingContext>,
    keys_state: HashMap<KeyCode, bool>,
}

// DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessengerCallbackData<'_>
fn debug_messenger_callback(
    message_severity: DebugUtilsMessageSeverity,
    message_type: DebugUtilsMessageType,
    callback_data: DebugUtilsMessengerCallbackData<'_>,
) {
    // https://github.com/gfx-rs/wgpu/pull/3627
    // This seems to not be a real error
    const VUID_VKSWAPCHAINCREATEINFOKHR_PNEXT_07781: i32 = 0x4C8929C1;
    if callback_data.message_id_number == VUID_VKSWAPCHAINCREATEINFOKHR_PNEXT_07781 {
        return;
    }

    let severity = if message_severity.intersects(DebugUtilsMessageSeverity::ERROR) {
        "error"
    } else if message_severity.intersects(DebugUtilsMessageSeverity::WARNING) {
        "warning"
    } else if message_severity.intersects(DebugUtilsMessageSeverity::INFO) {
        "information"
    } else if message_severity.intersects(DebugUtilsMessageSeverity::VERBOSE) {
        "verbose"
    } else {
        panic!("no-impl");
    };

    let ty = if message_type.intersects(DebugUtilsMessageType::GENERAL) {
        "general"
    } else if message_type.intersects(DebugUtilsMessageType::VALIDATION) {
        "validation"
    } else if message_type.intersects(DebugUtilsMessageType::PERFORMANCE) {
        "performance"
    } else {
        panic!("no-impl");
    };

    info!(
        "{} {} {}: {}",
        callback_data.message_id_name.unwrap_or("unknown"),
        ty,
        severity,
        callback_data.message
    );
}

fn select_physical_device(
    instance: Arc<Instance>,
    device_extensions: DeviceExtensions,
    event_loop: &impl HasDisplayHandle,
) -> Result<(Arc<PhysicalDevice>, u32), &'static str> {
    return instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|candidate_physical_device| {
            candidate_physical_device
                .supported_extensions()
                .contains(&device_extensions)
        })
        .filter_map(|candidate_physical_device| {
            candidate_physical_device
                .queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, queue)| {
                    queue.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && candidate_physical_device
                            .presentation_support(i as u32, event_loop)
                            .unwrap_or(false)
                })
                .map(|i| (candidate_physical_device, i as u32))
        })
        .min_by_key(|(candidate_physical_device, _)| {
            match candidate_physical_device.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            }
        })
        .ok_or("No suitable physical device found");
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        log::set_max_level(log::LevelFilter::Debug);
        log::set_logger(Box::leak(Box::new(logger::LoggerImplementation {}))).unwrap();
        let library = VulkanLibrary::new().unwrap();

        let required_extensions = InstanceExtensions {
            ext_debug_utils: true,
            ..Surface::required_extensions(event_loop).unwrap()
        };

        let required_layers = vec!["VK_LAYER_KHRONOS_validation".to_string()];

        // Validate that required layers are available
        let available_layers_names: Vec<_> = library
            .layer_properties()
            .unwrap()
            .map(|layer_properties| layer_properties.name().to_string())
            .collect();

        if let Some(not_found_layer_name) = required_layers.iter().find(|required_layer_name| {
            !available_layers_names.contains(&required_layer_name.to_string())
        }) {
            panic!(
                "Required validation layer: \"{}\" is missing.",
                not_found_layer_name
            );
        }

        // Now creating the instance.
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                // Enable enumerating devices that use non-conformant Vulkan implementations.
                // (e.g. MoltenVK)
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                enabled_layers: required_layers,
                ..Default::default()
            },
        )
        .unwrap();

        // After creating the instance we must register the debug callback.
        //
        // NOTE: If you let this debug_callback binding fall out of scope then the callback will stop
        // providing events.
        let _debug_callback = Arc::new(
            unsafe {
                DebugUtilsMessenger::new(
                    instance.clone(),
                    DebugUtilsMessengerCreateInfo {
                        message_severity: DebugUtilsMessageSeverity::ERROR
                            | DebugUtilsMessageSeverity::WARNING
                            | DebugUtilsMessageSeverity::INFO
                            | DebugUtilsMessageSeverity::VERBOSE,
                        message_type: DebugUtilsMessageType::GENERAL
                            | DebugUtilsMessageType::VALIDATION
                            | DebugUtilsMessageType::PERFORMANCE,
                        ..DebugUtilsMessengerCreateInfo::user_callback(
                            DebugUtilsMessengerCallback::new(debug_messenger_callback),
                        )
                    },
                )
            }
            .unwrap(),
        );

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            khr_shader_draw_parameters: true,
            ext_descriptor_indexing: true,
            ..DeviceExtensions::empty()
        };

        let device_features = DeviceFeatures {
            descriptor_binding_partially_bound: true,
            descriptor_binding_variable_descriptor_count: true,
            runtime_descriptor_array: true,
            shader_sampled_image_array_non_uniform_indexing: true,
            shader_draw_parameters: true,
            multi_draw_indirect: true,
            ..Default::default()
        };

        let (physical_device, queue_family_index) =
            select_physical_device(instance.clone(), device_extensions, event_loop).unwrap();

        info!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        info!("Queue {queue_family_index}");

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![
                    QueueCreateInfo {
                        queue_family_index,
                        // queues: vec![0.5, 0.1], // one queue for rendering and another for asset
                        //                         // loading
                        ..Default::default()
                    },
                    QueueCreateInfo {
                        queue_family_index: 1,
                        ..Default::default()
                    },
                ],
                enabled_features: device_features,
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let world = World::new();

        let (vertex_count, index_count) = count_vertices_and_indices_in_gltf_scene(
            "assets/meshes/Bistro_with_tangents_all.glb",
            //"assets/meshes/DamagedHelmet.glb",
        );

        let asset_database = Arc::new(RwLock::new(AssetDatabase::new(
            queues.next().unwrap(),
            memory_allocator.clone(),
            vertex_count as DeviceSize,
            index_count as DeviceSize,
        )));

        let irradiance_map_image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R16G16B16A16_SFLOAT,
                extent: [32, 32, 1],
                array_layers: 6,
                flags: ImageCreateFlags::CUBE_COMPATIBLE,
                usage: ImageUsage::TRANSFER_DST
                    | ImageUsage::SAMPLED
                    | ImageUsage::COLOR_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();
        info!("Created irradiance map image");

        let irradiance_map = ImageView::new(
            irradiance_map_image.clone(),
            ImageViewCreateInfo {
                view_type: ImageViewType::Cube,
                ..ImageViewCreateInfo::from_image(&irradiance_map_image)
            },
        )
        .unwrap();

        let prefiltered_environment_map_image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R16G16B16A16_SFLOAT,
                extent: [512, 512, 1],
                array_layers: 6,
                mip_levels: 5,
                flags: ImageCreateFlags::CUBE_COMPATIBLE,
                usage: ImageUsage::TRANSFER_DST
                    | ImageUsage::SAMPLED
                    | ImageUsage::COLOR_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();
        info!("Created prefiltered environment image");

        let prefiltered_environment_map = ImageView::new(
            prefiltered_environment_map_image.clone(),
            ImageViewCreateInfo {
                view_type: ImageViewType::Cube,
                ..ImageViewCreateInfo::from_image(&prefiltered_environment_map_image)
            },
        )
        .unwrap();

        let environment_brdf_lut_image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R16G16_SFLOAT,
                extent: [512, 512, 1],
                usage: ImageUsage::TRANSFER_DST
                    | ImageUsage::SAMPLED
                    | ImageUsage::COLOR_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();
        info!("Created environment brdf lut");

        let environment_brdf_lut = ImageView::new(
            environment_brdf_lut_image.clone(),
            ImageViewCreateInfo {
                view_type: ImageViewType::Dim2d,
                ..ImageViewCreateInfo::from_image(&environment_brdf_lut_image)
            },
        )
        .unwrap();

        let environment_map_id = asset_database
            .write()
            .unwrap()
            .add_cubemap_from_path([
                "assets/cubemaps/skybox/px.hdr",
                "assets/cubemaps/skybox/nx.hdr",
                "assets/cubemaps/skybox/py.hdr",
                "assets/cubemaps/skybox/ny.hdr",
                "assets/cubemaps/skybox/pz.hdr",
                "assets/cubemaps/skybox/nz.hdr",
            ])
            .unwrap();
        info!("Loaded skybox");

        let irradiance_map_renderer = ImageBasedLightingMapsGenerator::new(
            device.clone(),
            queue.clone(),
            memory_allocator.clone(),
            irradiance_map_image.format(),
        );

        irradiance_map_renderer.render_to_image(
            asset_database
                .read()
                .unwrap()
                .get_cubemap(environment_map_id.clone())
                .unwrap()
                .cubemap
                .clone(),
            irradiance_map_image.clone(),
            prefiltered_environment_map_image.clone(),
            environment_brdf_lut_image.clone(),
        );
        info!("Created ibl maps");

        let mut asset_database_write = asset_database.write().unwrap();
        world.set(EnvironmentCubemap {
            environment_map: environment_map_id,
            irradiance_map: asset_database_write
                .add_cubemap_from_raw(irradiance_map.clone())
                .unwrap(),
            prefiltered_environment_map: asset_database_write
                .add_cubemap_from_raw(prefiltered_environment_map.clone())
                .unwrap(),
            environment_brdf_lut: asset_database_write
                .add_cubemap_from_raw(environment_brdf_lut.clone())
                .unwrap(),
        });
        drop(asset_database_write);

        App {
            instance,
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            world,
            entities_to_add_queue: Arc::new(RwLock::new(vec![])),
            asset_database,
            _debug_callback,
            rendering_context: None,
            keys_state: HashMap::new(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.rendering_context.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(Window::default_attributes())
                    .unwrap(),
            );

            let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();
            let window_size = window.inner_size();

            let (swapchain, images) = {
                let surface_capabilities = self
                    .device
                    .physical_device()
                    .surface_capabilities(&surface, Default::default())
                    .unwrap();

                let (image_format, _) = self
                    .device
                    .physical_device()
                    .surface_formats(&surface, Default::default())
                    .unwrap()[0];

                Swapchain::new(
                    self.device.clone(),
                    surface.clone(),
                    SwapchainCreateInfo {
                        min_image_count: surface_capabilities.min_image_count.max(2),
                        image_format,
                        present_mode: PresentMode::Fifo,
                        image_extent: window_size.into(),
                        image_usage: ImageUsage::COLOR_ATTACHMENT,
                        composite_alpha: surface_capabilities
                            .supported_composite_alpha
                            .into_iter()
                            .next()
                            .unwrap(),
                        ..Default::default()
                    },
                )
                .unwrap()
            };

            let frames_in_flight = images.len();

            info!("FRAMES IN FLIGHT {frames_in_flight}");

            let swapchain_image_views: Vec<_> = images
                .iter()
                .map(|image| ImageView::new_default(image.clone()).unwrap())
                .collect();

            let renderer = Arc::new(RwLock::new(Renderer::new(
                self.device.clone(),
                self.queue.clone(),
                self.asset_database.clone(),
                self.memory_allocator.clone(),
                self.world.clone(),
                Format::R8G8B8A8_UNORM,
            )));

            let asset_database_clone = self.asset_database.clone();
            let entities_to_add_queue_clone = self.entities_to_add_queue.clone();
            std::thread::spawn(move || {
                load_gltf_scene(
                    "assets/meshes/Bistro_with_tangents_all.glb",
                    //"assets/meshes/DamagedHelmet.glb",
                    entities_to_add_queue_clone,
                    asset_database_clone.clone(),
                )
                .unwrap();
                info!("Loaded scene");
            });

            let gui = Gui::new(
                &event_loop,
                surface.clone(),
                self.queue.clone(),
                swapchain.image_format(),
                GuiConfig {
                    is_overlay: false,
                    ..GuiConfig::default()
                },
            );

            let mut user_interface = UserInterface::new(
                self.memory_allocator.clone(),
                renderer.clone(),
                gui,
                frames_in_flight,
            );

            user_interface.add_camera_view(Camera {
                position: Vec3::new(0.0, 0.0, 1.0),
                rotation: Quat::from_euler(EulerRot::YXZ, 0.0, 0.0, 0.0),
                fov: 90f32.to_radians(),
            });
            user_interface.add_log_view();
            user_interface.add_scene_tree(self.world.clone());

            let mut fences = vec![];
            for _ in 0..frames_in_flight {
                fences.push(None);
            }

            self.rendering_context = Some(RenderingContext {
                window,
                swapchain,
                swapchain_image_views,
                recreate_swapchain: false,
                fences,
                previous_fence_index: 0,
                renderer,
                start_frame_instant: Instant::now(),
                user_interface,
                frametime: Duration::ZERO,
            });
        }
    }

    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        match event {
            DeviceEvent::Key(RawKeyEvent {
                physical_key: PhysicalKey::Code(KeyCode::KeyQ),
                state: ElementState::Pressed,
            }) => event_loop.exit(),
            DeviceEvent::Key(RawKeyEvent {
                physical_key: PhysicalKey::Code(key_code),
                state,
            }) => {
                self.keys_state
                    .insert(key_code, state == ElementState::Pressed);
            }
            DeviceEvent::MouseMotion { delta } => {
                let rcx = self.rendering_context.as_mut().unwrap();
                if let Some(camera) = rcx.user_interface.get_focused_camera() {
                    let sensitivity = 0.5f32;

                    let yaw = Quat::from_rotation_y(-delta.0 as f32 * sensitivity * 0.01);
                    let pitch = Quat::from_rotation_x(-delta.1 as f32 * sensitivity * 0.01);

                    camera.rotation = yaw * camera.rotation * pitch;
                }
            }
            DeviceEvent::MouseWheel { delta: _delta } => {
                // TODO: zoom
            }
            _ => {}
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let rcx = self.rendering_context.as_mut().unwrap();

        if rcx.user_interface.get_focused_camera().is_some() {
            rcx.window.set_cursor_grab(CursorGrabMode::Confined).unwrap();
            rcx.window.set_cursor_visible(false);
        } else if rcx.user_interface.update(&event) {
            return;
        }


        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::KeyQ),
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } => {
                rcx.user_interface.unfocus_camera();
                rcx.window.set_cursor_grab(CursorGrabMode::None).unwrap();
                rcx.window.set_cursor_visible(true);
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                // rcx.is_mouse_captured = true;
                // rcx.window
                //     .set_cursor_grab(if rcx.is_mouse_captured {
                //         CursorGrabMode::Confined
                //     } else {
                //         CursorGrabMode::None
                //     })
                //     .unwrap();
                // rcx.window.set_cursor_visible(!rcx.is_mouse_captured);
            }
            WindowEvent::Resized(..) => {
                rcx.recreate_swapchain = true;
            }
            WindowEvent::RedrawRequested => {
                if let Some(camera) = rcx.user_interface.get_focused_camera() {
                    let speed = 5f32;
                    let forward = -(camera.rotation * Vec3::Z);
                    let right = camera.rotation * Vec3::X;
                    let mut direction = Vec3::ZERO;

                    if self
                        .keys_state
                        .get(&KeyCode::KeyW)
                        .cloned()
                        .unwrap_or(false)
                    {
                        direction += forward;
                    }
                    if self
                        .keys_state
                        .get(&KeyCode::KeyS)
                        .cloned()
                        .unwrap_or(false)
                    {
                        direction -= forward;
                    }
                    if self
                        .keys_state
                        .get(&KeyCode::KeyA)
                        .cloned()
                        .unwrap_or(false)
                    {
                        direction -= right;
                    }
                    if self
                        .keys_state
                        .get(&KeyCode::KeyD)
                        .cloned()
                        .unwrap_or(false)
                    {
                        direction += right;
                    }
                    if self
                        .keys_state
                        .get(&KeyCode::Space)
                        .cloned()
                        .unwrap_or(false)
                    {
                        direction += Vec3::Y;
                    }
                    if self
                        .keys_state
                        .get(&KeyCode::ShiftLeft)
                        .cloned()
                        .unwrap_or(false)
                    {
                        direction -= Vec3::Y;
                    }

                    if direction.length_squared() > 0.0 {
                        direction = direction.normalize();
                    }

                    camera.position += direction * speed * 0.01;
                }

                let mut entities_to_add_queue_write = self.entities_to_add_queue.write().unwrap();
                let old_entities_to_add_queue = std::mem::take(&mut *entities_to_add_queue_write);
                drop(entities_to_add_queue_write);
                for (name, transform, mesh, material) in old_entities_to_add_queue {
                    self.world
                        .entity_named(&name)
                        .add::<SceneEntity>()
                        .set(transform)
                        .set(mesh)
                        .set(material);
                }

                if rcx.recreate_swapchain {
                    let window_size = rcx.window.inner_size();
                    let images;

                    // Resize swapchain
                    (rcx.swapchain, images) = rcx
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..rcx.swapchain.create_info()
                        })
                        .expect("Failed to recreate swapchain");

                    // Create ImageViews for swapchain images
                    rcx.swapchain_image_views = images
                        .iter()
                        .map(|image| ImageView::new_default(image.clone()).unwrap())
                        .collect();

                    rcx.recreate_swapchain = false;
                }

                let (
                    image_index,
                    suboptimal,
                    swapchain_image_available_future
                ) = match acquire_next_image(
                    rcx.swapchain.clone(),
                    None,
                )
                .map_err(Validated::unwrap)
                {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

                if suboptimal {
                    rcx.recreate_swapchain = true;
                }

                let timer = ProfileTimer::start("cleanup_finished");
                let previous_future = match rcx.fences[rcx.previous_fence_index as usize].take() {
                    // Create a NowFuture
                    None => {
                        let mut now = sync::now(self.device.clone());
                        now.cleanup_finished();

                        now.boxed()
                    }
                    // Use the existing FenceSignalFuture
                    Some(mut fence) => {
                        fence.cleanup_finished();
                        fence.boxed()
                    }
                };
                drop(timer);

                let mut builder = AutoCommandBufferBuilder::primary(
                    self.command_buffer_allocator.clone(),
                    self.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                self.asset_database.write().unwrap()
                    .add_newly_loaded_meshes_to_main_buffers(&mut builder);

                rcx.user_interface.build(
                    image_index as usize,
                    &mut builder,
                    rcx.frametime,
                );

                let timer = ProfileTimer::start("build");
                let command_buffer = builder.build().unwrap();
                drop(timer);

                let timer = ProfileTimer::start("present");
                let future_3d_renderer = previous_future
                    .then_execute(self.queue.clone(), command_buffer)
                    .unwrap()
                    .then_signal_fence_and_flush()
                    .unwrap();
                drop(timer);

                let future_egui = rcx.user_interface.draw(
                    swapchain_image_available_future.join(future_3d_renderer),
                    &rcx.swapchain_image_views[image_index as usize],
                );

                let future = future_egui
                    .then_swapchain_present(
                        self.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            rcx.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                rcx.fences[image_index as usize] = match future.map_err(Validated::unwrap) {
                    Ok(value) => {
                        Some(value)
                    }
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        warn!("OutOfDate");
                        None
                    }
                    Err(e) => {
                        warn!("failed to flush future: {e}");
                        None
                    }
                };


                rcx.previous_fence_index = image_index;

                rcx.frametime = rcx.start_frame_instant.elapsed();
                rcx.start_frame_instant = Instant::now();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.rendering_context
            .as_ref()
            .unwrap()
            .window
            .request_redraw();
    }
}

