use vulkano::buffer::BufferContents;

#[derive(BufferContents, vulkano::pipeline::graphics::vertex_input::Vertex, Clone, Debug)]
#[repr(C)]
pub struct Vertex {
    #[format(R32G32B32_SFLOAT)]
    pub a_position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub a_normal: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub a_tangent: [f32; 3],
    #[format(R32G32_SFLOAT)]
    pub a_uv: [f32; 2],
}
