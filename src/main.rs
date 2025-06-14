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
mod renderer;
mod profile;

use assets::{database::AssetDatabase, loaders::gltf_scene_loader::load_gltf_scene};
use camera::Camera;
use ecs::components::{self, EnvironmentCubemap};
use flecs_ecs::prelude::*;
use glam::{EulerRot, Quat, Vec3};
use image_based_lighting_maps_generator::ImageBasedLightingMapsGenerator;
use renderer::Renderer;
use sdl2::{
    event::{Event, WindowEvent},
    keyboard::{Keycode, Scancode},
    mouse::MouseButton,
    video::Window,
    Sdl,
};
use std::sync::{Arc, RwLock};
use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags,
    }, format::Format, image::{
        view::{ImageView, ImageViewCreateInfo, ImageViewType},
        Image, ImageCreateFlags, ImageCreateInfo, ImageType, ImageUsage,
    }, instance::{
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger,
            DebugUtilsMessengerCallback, DebugUtilsMessengerCallbackData,
            DebugUtilsMessengerCreateInfo,
        },
        Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions,
    }, memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator}, VulkanLibrary
};

fn main() {
    let mut app = App::new();
    app.run();
}

struct App {
    _instance: Arc<Instance>,
    world: World,
    asset_database: Arc<RwLock<AssetDatabase>>,
    renderer: Renderer,
    camera: Camera,
    sdl_context: Sdl,
    _debug_callback: Arc<DebugUtilsMessenger>,
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

    println!(
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
    _window: Arc<Window>,
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
                .position(|(_i, queue)| {
                    queue.queue_flags.intersects(QueueFlags::GRAPHICS)
                    //&& candidate_physical_device.presentation_support(i as u32, &window).unwrap_or(false) // TODO: Make this work
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
    fn new() -> Self {
        let sdl_context = sdl2::init().unwrap();
        let video_subsystem = sdl_context.video().unwrap();
        let library = VulkanLibrary::new().unwrap();

        let window = Arc::new(
            video_subsystem
                .window("Window name", 1280, 720)
                .resizable()
                .vulkan()
                .build()
                .unwrap(),
        );

        let required_extensions = InstanceExtensions {
            ext_debug_utils: true,
            ..InstanceExtensions::from_iter(window.vulkan_instance_extensions().unwrap())
        };

        let required_layers = vec!["VK_LAYER_KHRONOS_validation".to_string()];

        // Validate that required layers are available
        let available_layers_names: Vec<String> = library
            .layer_properties()
            .unwrap()
            .map(|layer_properties| layer_properties.name().to_string())
            .collect();

        if let Some(not_found_layer_name) = required_layers
            .iter()
            .find(|required_layer_name| !available_layers_names.contains(required_layer_name))
        {
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
        let debug_callback = Arc::new(
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
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) =
            select_physical_device(instance.clone(), device_extensions, window.clone()).unwrap();

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let world = World::new();

        let asset_database = Arc::new(RwLock::new(
            AssetDatabase::new(queue.clone(), memory_allocator.clone())
        ));

        let renderer = Renderer::new(
            instance.clone(),
            window.clone(),
            device.clone(),
            queue.clone(),
            asset_database.clone(),
            memory_allocator.clone(),
            world.clone(),
        );
        println!("Initialized renderer");

        load_gltf_scene(
            "assets/meshes/Bistro_with_tangents_all.glb",
            world.clone(),
            asset_database.clone(),
        ).unwrap();
        println!("Loaded scene");

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
        println!("Created irradiance map image");

        let irradiance_map = ImageView::new(
            irradiance_map_image.clone(),
            ImageViewCreateInfo {
                view_type: ImageViewType::Cube,
                ..ImageViewCreateInfo::from_image(&irradiance_map_image.clone())
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
        println!("Created prefiltered environment image");

        let prefiltered_environment_map = ImageView::new(
            prefiltered_environment_map_image.clone(),
            ImageViewCreateInfo {
                view_type: ImageViewType::Cube,
                ..ImageViewCreateInfo::from_image(&prefiltered_environment_map_image.clone())
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
        println!("Created environment brdf lut");

        let environment_brdf_lut = ImageView::new(
            environment_brdf_lut_image.clone(),
            ImageViewCreateInfo {
                view_type: ImageViewType::Dim2d,
                ..ImageViewCreateInfo::from_image(&environment_brdf_lut_image.clone())
            },
        )
        .unwrap();

        let environment_map_id = asset_database
            .write().unwrap()
            .add_cubemap_from_path(
                [
                    "assets/cubemaps/skybox/px.hdr",
                    "assets/cubemaps/skybox/nx.hdr",
                    "assets/cubemaps/skybox/py.hdr",
                    "assets/cubemaps/skybox/ny.hdr",
                    "assets/cubemaps/skybox/pz.hdr",
                    "assets/cubemaps/skybox/nz.hdr",
                ],
            )
            .unwrap();
        println!("Loaded skybox");

        let irradiance_map_renderer = ImageBasedLightingMapsGenerator::new(
            device.clone(),
            queue.clone(),
            memory_allocator.clone(),
            irradiance_map_image.format(),
        );

        irradiance_map_renderer.render_to_image(
            asset_database.read().unwrap()
                .get_cubemap(environment_map_id.clone())
                .unwrap().cubemap.clone(),
            irradiance_map_image.clone(),
            prefiltered_environment_map_image.clone(),
            environment_brdf_lut_image.clone(),
        );
        println!("Created ibl maps");

        let mut asset_database_write = asset_database.write().unwrap();
        world.set(EnvironmentCubemap {
            environment_map: environment_map_id,
            irradiance_map: asset_database_write
                .add_cubemap_from_raw(irradiance_map.clone()).unwrap(),
            prefiltered_environment_map: asset_database_write
                .add_cubemap_from_raw(prefiltered_environment_map.clone()).unwrap(),
            environment_brdf_lut: asset_database_write
                .add_cubemap_from_raw(environment_brdf_lut.clone()).unwrap(),
        });
        drop(asset_database_write);

        App {
            _instance: instance,
            world,
            asset_database,
            camera: Camera {
                position: Vec3::new(0.0, 0.0, 1.0),
                rotation: Quat::from_euler(EulerRot::YXZ, 0.0, 0.0, 0.0),
                fov: 90f32.to_radians(),
            },
            sdl_context,
            _debug_callback: debug_callback,
            renderer,
        }
    }

    fn run(&mut self) {
        let mut event_pump = self.sdl_context.event_pump().unwrap();
        let mut captured = false;

        'running: loop {
            for event in event_pump.poll_iter() {
                match event {
                    Event::Quit { .. }
                    | Event::KeyDown {
                        keycode: Some(Keycode::Q),
                        ..
                    } => {
                        break 'running;
                    }
                    Event::KeyDown {
                        keycode: Some(Keycode::Escape),
                        ..
                    } => {
                        captured = !captured;
                        self.sdl_context.mouse().set_relative_mouse_mode(captured);
                    }
                    Event::MouseButtonDown {
                        mouse_btn: MouseButton::Left,
                        ..
                    } => {
                        captured = true;
                        self.sdl_context.mouse().set_relative_mouse_mode(captured);
                    }
                    Event::MouseMotion {
                        timestamp: _timestamp,
                        window_id: _window_id,
                        which: _which,
                        mousestate: _mousestate,
                        x: _x,
                        y: _y,
                        xrel,
                        yrel,
                    } if captured => {
                        let sensitivity = 0.5f32;

                        let yaw = Quat::from_rotation_y(-xrel as f32 * sensitivity * 0.01);
                        let pitch = Quat::from_rotation_x(-yrel as f32 * sensitivity * 0.01);

                        self.camera.rotation = yaw * self.camera.rotation * pitch;
                    }
                    Event::Window {
                        win_event: WindowEvent::Resized(..),
                        ..
                    } => {
                        //self.rcx.recreate_swapchain = true;
                    }
                    _ => {}
                }
            }

            if captured {
                let keyboard_state = event_pump.keyboard_state();
                let speed = 5f32;
                let forward = -(self.camera.rotation * Vec3::Z);
                let right = self.camera.rotation * Vec3::X;
                let mut direction = Vec3::ZERO;

                if keyboard_state.is_scancode_pressed(Scancode::W) {
                    direction += forward;
                }
                if keyboard_state.is_scancode_pressed(Scancode::S) {
                    direction -= forward;
                }
                if keyboard_state.is_scancode_pressed(Scancode::A) {
                    direction -= right;
                }
                if keyboard_state.is_scancode_pressed(Scancode::D) {
                    direction += right;
                }
                if keyboard_state.is_scancode_pressed(Scancode::Space) {
                    direction += Vec3::Y;
                }
                if keyboard_state.is_scancode_pressed(Scancode::LShift) {
                    direction -= Vec3::Y;
                }

                if direction.length_squared() > 0.0 {
                    direction = direction.normalize();
                }
                self.camera.position += direction * speed * 0.01;

                if keyboard_state.is_scancode_pressed(Scancode::Right) {
                    self.world
                        .lookup("DamagedHelmet")
                        .get::<&mut components::Transform>(|transform| {
                            transform.rotation =
                                Quat::from_rotation_y(1f32.to_radians()) * transform.rotation;
                        });
                }
                if keyboard_state.is_scancode_pressed(Scancode::Left) {
                    self.world
                        .lookup("DamagedHelmet")
                        .get::<&mut components::Transform>(|transform| {
                            transform.rotation =
                                Quat::from_rotation_y(-1f32.to_radians()) * transform.rotation;
                        });
                }
            }


            //self.camera.position += self.camera_move_state.speed;

            self.renderer.draw(&self.camera);

            //::std::thread::sleep(::std::time::Duration::new(0, 1_000_000_000u32 / 60));
        }
    }
}
