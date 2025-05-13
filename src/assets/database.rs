use std::collections::HashMap;

use super::mesh::Mesh;

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct MeshId(u32);
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct TextureId(u32);
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct MaterialId(u32);

pub struct AssetDatabase {
    meshes: HashMap<MeshId, Mesh>,
    
}

impl AssetDatabase {
    pub fn new() -> Self {
        Self {
            meshes: HashMap::new(),
        }
    }

    pub fn add_mesh(&mut self, mesh: Mesh) -> MeshId {
        let mut mesh_id = MeshId(1);
        while self.meshes.contains_key(&mesh_id) {
            mesh_id.0 += 1;
        }
        self.meshes.insert(mesh_id.clone(), mesh);

        mesh_id
    }

    pub fn get_mesh(&self, mesh_id: &MeshId) -> Option<&Mesh> {
        self.meshes.get(mesh_id)
    }
}
