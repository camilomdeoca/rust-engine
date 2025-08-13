use egui::Response;
use glam::{Vec2, Vec3};
use smallvec::SmallVec;

const COMPONENT_NAMES: [&str; 4] = ["x: ", "y: ", "z: ", "w: "];

// macro_rules! impl_vec_edit {
//     ($fn_name:ident, $vec_type:ty, $length: literal) => {
//         pub fn $fn_name(ui: &mut egui::Ui, vec: &mut $vec_type, label: &str) {
//             ui.horizontal(|ui| {
//                 ui.label(label);
//                 for i in 0..$length {
//                     ui.add(
//                         egui::DragValue::new(&mut vec[i])
//                             .speed(0.1)
//                             .prefix(COMPONENT_NAMES[i]),
//                     );
//                 }
//             });
//         }
//     };
// }

// impl_vec_edit!(edit_vec2_ui, glam::Vec2, 2);
// impl_vec_edit!(edit_vec3_ui, glam::Vec3, 3);
// impl_vec_edit!(edit_vec4_ui, glam::Vec4, 4);
// impl_vec_edit!(edit_uvec2_ui, glam::UVec2, 2);
// impl_vec_edit!(edit_uvec3_ui, glam::UVec3, 3);
// impl_vec_edit!(edit_uvec4_ui, glam::UVec4, 4);

pub trait CanBeEditedAsGlamVec<T> {
    const LEN: usize;
    fn get_mut_component_at(&mut self, index: usize) -> &mut T;
}

macro_rules! impl_edit_ui_for_glam_vector {
    ($component_type:ty, $vec_type: ty, $length: literal) => {
        impl CanBeEditedAsGlamVec<$component_type> for $vec_type {
            const LEN: usize = $length;

            fn get_mut_component_at(&mut self, index: usize) -> &mut $component_type {
                &mut self[index]
            }
        }
    };
}

impl_edit_ui_for_glam_vector!(f32, Vec2, 2);
impl_edit_ui_for_glam_vector!(f32, Vec3, 3);

pub fn edit_ui<T, V>(ui: &mut egui::Ui, vec: &mut V) -> Response
where
    V: CanBeEditedAsGlamVec<T>,
    T: egui::emath::Numeric,
{
    ui.horizontal(|ui| {
        for i in 0..V::LEN {
            let label = ui.label(COMPONENT_NAMES[i]);
            ui.add(egui::DragValue::new(vec.get_mut_component_at(i)).speed(0.1))
                .labelled_by(label.id);
        }
    })
    .response
}

// pub fn edit_vec_of_u32(ui: &mut egui::Ui, vec: &mut Vec<u32>, label: &str) {
//     let mut to_remove_index = None;
//     ui.horizontal(|ui| {
//         ui.label(label);
//         for (i, elem) in vec.iter_mut().enumerate() {
//             ui.horizontal(|ui| {
//                 ui.add(egui::DragValue::new(elem).speed(1));
//                 if ui.button("X").clicked() {
//                     to_remove_index = Some(i);
//                 }
//             });
//         }
//         if ui.button("+").clicked() {
//             vec.push(0);
//         }
//     });
//     if let Some(to_remove_index) = to_remove_index {
//         vec.remove(to_remove_index);
//     }
// }

pub trait CanBeEditedAsVec<T> {
    fn len(&self) -> usize;
    fn get_mut(&mut self, index: usize) -> &mut T;
    fn push(&mut self, value: T);
    fn remove(&mut self, index: usize);
}

impl<T> CanBeEditedAsVec<T> for Vec<T> {
    fn len(&self) -> usize {
        self.len()
    }

    fn get_mut(&mut self, index: usize) -> &mut T {
        &mut self[index]
    }

    fn push(&mut self, value: T) {
        self.push(value);
    }

    fn remove(&mut self, index: usize) {
        self.remove(index);
    }
}

impl<T, const L: usize> CanBeEditedAsVec<T> for SmallVec<[T; L]> {
    fn len(&self) -> usize {
        self.len()
    }

    fn get_mut(&mut self, index: usize) -> &mut T {
        &mut self[index]
    }

    fn push(&mut self, value: T) {
        self.push(value);
    }

    fn remove(&mut self, index: usize) {
        self.remove(index);
    }
}

pub fn edit_vec_ui<T, V>(ui: &mut egui::Ui, vec: &mut V, default: &T) -> Response
where
    V: CanBeEditedAsVec<T>,
    T: egui::emath::Numeric,
{
    let mut to_remove_index = None;

    let response = ui.horizontal(|ui| {
        for i in 0..vec.len() {
            let elem = vec.get_mut(i);
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                ui.add(egui::DragValue::new(elem).speed(1));
                if ui.button("X").clicked() {
                    to_remove_index = Some(i);
                }
            });
        }

        if ui.button("+").clicked() {
            vec.push(default.clone());
        }
    });

    if let Some(to_remove_index) = to_remove_index {
        vec.remove(to_remove_index);
    }

    response.response
}

pub fn edit_array_ui<T>(ui: &mut egui::Ui, slice: &mut [T]) -> Response
where
    T: egui::emath::Numeric,
{
    let response = ui.horizontal(|ui| {
        for elem in slice {
            ui.add(egui::DragValue::new(elem).speed(1));
        }
    });

    response.response
}
