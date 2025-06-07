use std::{collections::HashMap, fs, path::Path, sync::Arc};

use flecs_ecs::core::World;
use glam::{Quat, Vec3};
use gltf::{buffer::Data, image::Source, mesh::util::ReadIndices, Gltf, Texture};
use vulkano::{
    command_buffer::allocator::CommandBufferAllocator, device::Queue, format::Format,
    image::view::ImageView, memory::allocator::MemoryAllocator,
};

use crate::{
    assets::{loaders::texture_loader::load_texture_from_buffer, vertex::Vertex},
    ecs::components::{Material, Mesh, Transform},
};

use super::mesh_loader::load_mesh_from_buffers;

fn load_texture_from_gltf_texture(
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    queue: Arc<Queue>,
    gltf_texture: &Texture,
    gltf_buffers: &Vec<Data>,
) -> Arc<ImageView> {
    match gltf_texture.source().source() {
        Source::View { view, mime_type } => {
            let start = view.offset();
            let end = view.offset() + view.length();
            let buffer = &gltf_buffers[view.buffer().index()][start..end];
            let image = image::load_from_memory_with_format(
                buffer,
                image::ImageFormat::from_mime_type(mime_type).unwrap(),
            )
            .unwrap()
            .into_rgba8();
            load_texture_from_buffer(
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                queue.clone(),
                Format::R8G8B8A8_UNORM,
                image.dimensions().into(),
                &image,
            )
            .unwrap()
        }
        Source::Uri { uri, mime_type } => todo!(),
    }
}

pub fn load_gltf_scene(
    path: impl AsRef<Path>,
    world: &World,
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    queue: Arc<Queue>,
) -> Result<(), String> {
    let path = path.as_ref();
    let file_data = fs::read(&path).unwrap();
    let gltf = gltf::Gltf::from_slice(&file_data).unwrap();
    println!("Importing {:?} ({} meshes)", path, gltf.meshes().len());
    let base_path = path.parent().unwrap_or_else(|| Path::new("./"));
    let gltf_buffers = gltf::import_buffers(&gltf.document, Some(base_path), gltf.blob).unwrap();

    let mut textures = <HashMap<_, Arc<ImageView>>>::default();

    //let (gltf, gltf_buffers, _images) = gltf::import(&path).unwrap();

    for node in gltf.document.nodes() {
        if node.mesh().is_none() {
            continue;
        }
        let mesh = node.mesh().unwrap();

        println!(
            "    [{}/{}]: Mesh {}",
            node.index(),
            gltf.document.nodes().len(),
            node.name().unwrap_or("UNNAMED")
        );

        for primitive in mesh.primitives() {
            println!(
                "        Material {}",
                primitive.material().name().unwrap(),
            );
            let reader = primitive.reader(|buffer| Some(&gltf_buffers[buffer.index()]));

            let positions_iter = reader.read_positions().unwrap();
            let normals_iter = reader.read_normals().unwrap();
            let tangents_iter = reader.read_tangents();
            // if tangents_iter.is_none() {
            //     println!("        HAS_NO_TANGENTS");
            //     continue;
            // }
            let tangents_iter = tangents_iter.unwrap();
            let uvs_iter = reader.read_tex_coords(0).unwrap().into_f32();

            let mut vertices = Vec::with_capacity(positions_iter.len());

            for (((a_position, a_normal), a_tangent), a_uv) in positions_iter
                .zip(normals_iter)
                .zip(tangents_iter)
                .zip(uvs_iter)
            {
                vertices.push(Vertex {
                    a_position,
                    a_normal,
                    a_tangent: [a_tangent[0], a_tangent[1], a_tangent[2]],
                    a_uv,
                });
            }
            assert_eq!(vertices.capacity(), vertices.len());

            let indices_reader = reader.read_indices().unwrap();

            let indices: Vec<_> = match indices_reader {
                ReadIndices::U8(iter) => iter.map(|index| index.into()).collect(),
                ReadIndices::U16(iter) => iter.map(|index| index.into()).collect(),
                ReadIndices::U32(iter) => iter.collect(),
            };

            let (vertex_buffer, index_buffer) = load_mesh_from_buffers(
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                queue.clone(),
                vertices,
                indices,
            )
            .unwrap();

            let mut get_texture_from_gltf_texture = |gltf_texture: &gltf::Texture| -> Arc<ImageView> {
                if let Some(texture) = textures.get(&gltf_texture.index()) {
                    texture.clone()
                } else {
                    let texture = load_texture_from_gltf_texture(
                        memory_allocator.clone(),
                        command_buffer_allocator.clone(),
                        queue.clone(),
                        &gltf_texture,
                        &gltf_buffers,
                    );
                    textures.insert(gltf_texture.index(), texture.clone());

                    texture
                }
            };

            let gltf_material = primitive.material();

            let color_factor = gltf_material.pbr_metallic_roughness().base_color_factor().into();
            let diffuse = gltf_material
                .pbr_metallic_roughness()
                .base_color_texture()
                .map(|texture_info| get_texture_from_gltf_texture(&texture_info.texture()));

            let metallic_factor = gltf_material.pbr_metallic_roughness().metallic_factor().into();
            let roughness_factor = gltf_material.pbr_metallic_roughness().roughness_factor().into();
            let metallic_roughness = gltf_material
                .pbr_metallic_roughness()
                .metallic_roughness_texture()
                .map(|texture_info| get_texture_from_gltf_texture(&texture_info.texture()));

            let ambient_oclussion = gltf_material
                .occlusion_texture()
                .map(|occlusion_texture| get_texture_from_gltf_texture(&occlusion_texture.texture()));

            let emissive_factor = gltf_material.emissive_factor().into();
            let emissive = gltf_material
                .emissive_texture()
                .map(|emissive_texture| get_texture_from_gltf_texture(&emissive_texture.texture()));

            let normal = gltf_material
                .normal_texture()
                .map(|normal_texture| get_texture_from_gltf_texture(&normal_texture.texture()));

            let (translation, rotation, scale) = node.transform().decomposed();

            let name = mesh.name().unwrap().to_string() + primitive.material().name().unwrap();

            world
                .entity_named(&name)
                .set(Transform {
                    translation: Vec3::from_array(translation),
                    rotation: Quat::from_array(rotation)
                        * Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
                    scale: Vec3::from_array(scale),
                })
                .set(Mesh {
                    vertex_buffer,
                    index_buffer,
                })
                .set(Material {
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
        }
    }

    Ok(())
}
