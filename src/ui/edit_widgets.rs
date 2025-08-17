use egui::{Response, Widget};
use glam::{EulerRot, Quat, Vec3};

const COMPONENT_NAMES: [&str; 4] = ["x:", "y:", "z:", "w:"];

macro_rules! impl_edit_ui_for_glam_vector {
    ($widget_name: ident, $vec_type: ty, $length: literal) => {
        pub struct $widget_name<'a> {
            value: &'a mut $vec_type,
        }

        impl<'a> $widget_name<'a> {
            pub fn new(value: &'a mut $vec_type) -> Self {
                Self { value }
            }
        }

        impl Widget for $widget_name<'_> {
            fn ui(self, ui: &mut egui::Ui) -> Response {
                ui.horizontal(|ui| {
                    for i in 0..$length {
                        let label = ui.label(COMPONENT_NAMES[i]);
                        ui.add(egui::DragValue::new(&mut self.value[i]).speed(0.1))
                            .labelled_by(label.id);
                    }
                })
                .response
            }
        }
    };
}

impl_edit_ui_for_glam_vector!(Vec3DragEdit, Vec3, 3);

pub struct QuatDragEditAsEulerAngles<'a> {
    value: &'a mut Quat,
}

impl<'a> QuatDragEditAsEulerAngles<'a> {
    pub fn new(value: &'a mut Quat) -> Self {
        Self { value }
    }
}

const EULER_ANGLES_COMPONENTS_NAMES: [&str; 3] = ["pitch:", "yaw:", "roll:"];

impl Widget for QuatDragEditAsEulerAngles<'_> {
    fn ui(self, ui: &mut egui::Ui) -> Response {
        ui.horizontal(|ui| {
            let mut euler =
                Into::<[_; 3]>::into(self.value.to_euler(EulerRot::YXZ)).map(|v| v.to_degrees());

            for (i, elem) in euler.iter_mut().enumerate() {
                let label = ui.label(EULER_ANGLES_COMPONENTS_NAMES[i]);
                ui.add(egui::DragValue::new(elem).speed(0.1))
                    .labelled_by(label.id);
            }

            euler = euler.map(|v| v.to_radians());

            *self.value = Quat::from_euler(EulerRot::YXZ, euler[0], euler[1], euler[2]);
        })
        .response
    }
}

// pub fn edit_vec_ui<T, V>(ui: &mut egui::Ui, vec: &mut V, default: &T) -> Response
// where
//     V: CanBeEditedAsVec<T> + IndexMut<usize>,
//     V: Index<usize, Output = T>,
//     T: egui::emath::Numeric,
// {
//     let mut to_remove_index = None;
//
//     let response = ui.horizontal(|ui| {
//         for i in 0..vec.len() {
//             ui.horizontal(|ui| {
//                 ui.spacing_mut().item_spacing.x = 0.0;
//                 ui.add(egui::DragValue::new(&mut vec[i]).speed(1));
//                 if ui.button("X").clicked() {
//                     to_remove_index = Some(i);
//                 }
//             });
//         }
//
//         if ui.button("+").clicked() {
//             vec.push(default.clone());
//         }
//     });
//
//     if let Some(to_remove_index) = to_remove_index {
//         vec.remove(to_remove_index);
//     }
//
//     response.response
// }

pub struct ArrayDragEdit<'a, T: egui::emath::Numeric> {
    value: &'a mut [T],
}

impl<'a, T: egui::emath::Numeric> ArrayDragEdit<'a, T> {
    pub fn new(value: &'a mut [T]) -> Self {
        Self { value }
    }
}

impl<T: egui::emath::Numeric> Widget for ArrayDragEdit<'_, T> {
    fn ui(self, ui: &mut egui::Ui) -> Response {
        let response = ui.horizontal(|ui| {
            for elem in self.value {
                ui.add(egui::DragValue::new(elem).speed(1));
            }
        });

        response.response
    }
}
