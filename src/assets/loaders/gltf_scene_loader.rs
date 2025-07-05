use std::{collections::HashMap, fs, path::Path, sync::{Arc, RwLock}};

use glam::{Quat, Vec3};
use gltf::{buffer::Data, image::Source, mesh::util::ReadIndices, Texture};
use log::info;
use vulkano::format::Format;

use crate::{
    assets::{database::{AssetDatabase, MaterialId, MeshId, TextureId}, vertex::Vertex},
    ecs::components::{MaterialComponent, MeshComponent, Transform},
};

pub fn count_vertices_and_indices_in_gltf_scene(path: impl AsRef<Path>) -> (usize, usize) {
    let path = path.as_ref();
    let file_data = fs::read(&path).unwrap();
    let gltf = gltf::Gltf::from_slice(&file_data).unwrap();
    count_vertices_and_indices_in_gltf_document(&gltf.document)
}

fn count_vertices_and_indices_in_gltf_document(document: &gltf::Document) -> (usize, usize) {
    let mut vertex_count = 0;
    let mut index_count = 0;
    for node in document.nodes() {
        if node.mesh().is_none() {
            continue;
        }

        for primitive in node.mesh().unwrap().primitives() {
            vertex_count += primitive.get(&gltf::Semantic::Positions).unwrap().count();
            index_count += primitive.indices().unwrap().count();
        }
    }

    (vertex_count, index_count)
}

fn load_texture_from_gltf_texture(
    asset_database: Arc<RwLock<AssetDatabase>>,
    gltf_texture: &Texture,
    gltf_buffers: &Vec<Data>,
) -> TextureId {
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
            asset_database.write().unwrap().add_texture_from_buffer(
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
    added_entities: Arc<RwLock<Vec<(String, Transform, MeshComponent, MaterialComponent)>>>,
    asset_database: Arc<RwLock<AssetDatabase>>,
) -> Result<(), String> {
    let path = path.as_ref();
    let file_data = fs::read(&path).unwrap();
    let gltf = gltf::Gltf::from_slice(&file_data).unwrap();
    info!("Importing {:?} ({} meshes)", path, gltf.meshes().len());
    let base_path = path.parent().unwrap_or_else(|| Path::new("./"));
    let gltf_buffers = gltf::import_buffers(&gltf.document, Some(base_path), gltf.blob).unwrap();

    let mut textures = <HashMap<_, TextureId>>::default();
    let mut materials = <HashMap<_, MaterialId>>::default();
    let mut meshes = <HashMap<_, MeshId>>::default();

    let (vertex_count_2, index_count_2) = count_vertices_and_indices_in_gltf_document(&gltf.document);

    let mut vertex_count = 0;
    let mut index_count = 0;

    //let (gltf, gltf_buffers, _images) = gltf::import(&path).unwrap();

    for node in gltf.document.nodes() {
        if node.mesh().is_none() {
            continue;
        }
        let mesh = node.mesh().unwrap();

        info!(
            "    [{}/{}]: Mesh {}",
            node.index(),
            gltf.document.nodes().len(),
            node.name().unwrap_or("UNNAMED")
        );

        for primitive in mesh.primitives() {
            info!(
                "        Material {} {}",
                primitive.material().name().unwrap(),
                if materials.contains_key(&primitive.material().index().unwrap()) {
                    "(Repeated)"
                } else {
                    ""
                },
            );
            let mesh_id = if let Some(mesh_id) = meshes.get(&(mesh.index(), primitive.index())) {
                panic!("REPEATED MESH YAY"); // Change to println! once confirmed it happens
                mesh_id.clone()
            } else {
                let reader = primitive.reader(|buffer| Some(&gltf_buffers[buffer.index()]));

                let positions_iter = reader.read_positions().unwrap();
                vertex_count += positions_iter.len();
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

                index_count += indices.len();

                let mut asset_database_write = asset_database.write().unwrap();
                let mesh_id = asset_database_write.add_mesh_from_buffers(vertices, indices).unwrap();
                
                meshes.insert((mesh.index(), primitive.index()), mesh_id.clone());

                mesh_id
            };

            let mut get_texture_from_gltf_texture = |gltf_texture: &gltf::Texture| -> TextureId {
                if let Some(texture_id) = textures.get(&gltf_texture.index()) {
                    panic!("REPEATED TEXTURE YAY"); // Change to println! once confirmed it happens
                    texture_id.clone()
                } else {
                    let texture = load_texture_from_gltf_texture(
                        asset_database.clone(),
                        &gltf_texture,
                        &gltf_buffers,
                    );
                    textures.insert(gltf_texture.index(), texture.clone());

                    texture
                }
            };

            let gltf_material = primitive.material();
            let material_id = if let Some(material_id) = materials.get(&gltf_material.index().unwrap()) {
                material_id.clone()
            } else {
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

                let mut asset_database_write = asset_database.write().unwrap();
                let material_id = asset_database_write.add_material(
                    color_factor,
                    diffuse,
                    metallic_factor,
                    roughness_factor,
                    metallic_roughness,
                    ambient_oclussion,
                    emissive_factor,
                    emissive,
                    normal,
                ).unwrap();
                drop(asset_database_write);

                materials.insert(gltf_material.index().unwrap(), material_id.clone());

                material_id
            };

            let (translation, rotation, scale) = node.transform().decomposed();

            let name = mesh.name().unwrap().to_string() + primitive.material().name().unwrap();

            added_entities.write().unwrap().push((
                name,
                Transform {
                    translation: Vec3::from_array(translation),
                    rotation: Quat::from_array(rotation)
                        * Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
                    scale: Vec3::from_array(scale),
                },
                MeshComponent {
                    mesh_id,
                },
                MaterialComponent {
                    material_id,
                },
            ));
        }
    }

    assert_eq!(vertex_count, vertex_count_2);
    assert_eq!(index_count, index_count_2);

    Ok(())
}
