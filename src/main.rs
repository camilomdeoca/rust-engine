// Welcome to the triangle example!
//
// This is the only example that is entirely detailed. All the other examples avoid code
// duplication by using helper functions.
//
// This example assumes that you are already more or less familiar with graphics programming and
// that you want to learn Vulkan. This means that for example it won't go into details about what a
// vertex or a shader is.

mod assets;
mod ecs;
mod camera;
mod renderer;

use std::{path::Path, sync::Arc};
use assets::{database::AssetDatabase, Mesh};
use camera::Camera;
use ecs::components;
use flecs_ecs::prelude::*;
use glam::{EulerRot, Quat, Vec3};
use renderer::Renderer;
use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType}, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags
    }, instance::{debug::{DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger, DebugUtilsMessengerCallback, DebugUtilsMessengerCallbackData, DebugUtilsMessengerCreateInfo}, Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions}, memory::allocator::StandardMemoryAllocator, VulkanLibrary
};
use sdl2::{event::{Event, WindowEvent}, keyboard::{Keycode, Scancode}, mouse::MouseButton, video::Window, Sdl};

fn main() {
    let mut app = App::new();
    app.run();
}

struct App {
    instance: Arc<Instance>,
    world: World,
    asset_database: AssetDatabase,
    renderer: Renderer,
    camera: Camera,
    sdl_context: Sdl,
    _debug_callback: Arc<DebugUtilsMessenger>,
}

// DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessengerCallbackData<'_>
fn debug_messenger_callback(
    message_severity: DebugUtilsMessageSeverity,
    message_type: DebugUtilsMessageType,
    callback_data: DebugUtilsMessengerCallbackData<'_>
) {
    // https://github.com/gfx-rs/wgpu/pull/3627
    // This seems to not be a real error
    const VUID_VKSWAPCHAINCREATEINFOKHR_PNEXT_07781: i32 = 0x4C8929C1;
    if callback_data.message_id_number == VUID_VKSWAPCHAINCREATEINFOKHR_PNEXT_07781 {
        return;
    }

    let severity = if message_severity
        .intersects(DebugUtilsMessageSeverity::ERROR)
    {
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
            candidate_physical_device.supported_extensions().contains(&device_extensions)
        })
        .filter_map(|candidate_physical_device| {
            candidate_physical_device.queue_family_properties()
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
                .window("Window name", 640, 480)
                .resizable()
                .vulkan()
                .build()
                .unwrap()
        );

        let required_extensions = InstanceExtensions {
            ext_debug_utils: true,
            ..InstanceExtensions::from_iter(window.vulkan_instance_extensions().unwrap())
        };

        let required_layers = vec![
            "VK_LAYER_KHRONOS_validation".to_string()
        ];

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
            panic!("Required validation layer: \"{}\" is missing.", not_found_layer_name);
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
            .unwrap()
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

        let renderer = Renderer::new(
            instance.clone(),
            window.clone(),
            device.clone(),
            queue.clone(),
            memory_allocator.clone(),
        );

        let world = World::new();
        let mut asset_database = AssetDatabase::new();

        let damaged_helmet_mesh_id = asset_database.add_mesh(Mesh::from_path(
            memory_allocator.clone(),
            Path::new("/home/camilo/cosas/ruest/hello_world/DamagedHelmet.gltf")
        ).unwrap());
        let teapot_mesh_id = asset_database.add_mesh(Mesh::from_path(
            memory_allocator.clone(),
            Path::new("/home/camilo/cosas/ruest/hello_world/teapot.gltf")
        ).unwrap());

        world.entity_named("DamagedHelmet")
            .set(components::Transform {
                translation: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            })
            .set(components::Mesh { id: damaged_helmet_mesh_id });
        world.entity_named("Teapot")
            .set(components::Transform {
                translation: Vec3::new(0.0, 4.0, 0.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            })
            .set(components::Mesh { id: teapot_mesh_id });

        App {
            instance,
            world,
            asset_database,
            camera: Camera {
                position: Vec3::new(0.0, 0.0, 1.0),
                rotation: Quat::from_euler(EulerRot::YXZ, 0.0, 0.0, 0.0),
                fov: std::f32::consts::FRAC_PI_2,
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
            }

            //self.camera.position += self.camera_move_state.speed;

            self.renderer.draw(&self.camera, &self.world, &self.asset_database);

            ::std::thread::sleep(::std::time::Duration::new(0, 1_000_000_000u32 / 60));
        }
    }
}

