//! Example displaying a simple Mesh.

// TODO(#6330): remove unwrap()
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use glam::{UVec3, Vec2, Vec3};
use re_renderer::{
    Color32, DebugLabel, PickingLayerId, RenderContext, Rgba32Unmul,
    mesh::{CpuMesh, GpuMesh},
    renderer::GpuMeshInstance,
    view_builder::{self, Projection, ViewBuilder},
};

mod framework;

struct ExampleMesh {
    pub mesh_instances: Vec<GpuMeshInstance>,
}

impl framework::Example for ExampleMesh {
    fn title() -> &'static str {
        "Simple Mesh"
    }

    fn new(re_ctx: &RenderContext) -> Self {
        let MeshData {
            indices,
            positions,
            uvs,
            colors,
            normals,
        } = build_cube_mesh_data();

        let material = re_renderer::mesh::Material {
            label: "ground_mat".into(),
            index_range: 0..(indices.len() * 3) as u32,
            albedo: re_ctx
                .texture_manager_2d
                .white_texture_unorm_handle()
                .clone(),
            albedo_factor: re_renderer::Rgba::WHITE,
        };
        let cpu_mesh = CpuMesh {
            label: DebugLabel::from("ground".to_string()),
            triangle_indices: indices,
            vertex_positions: positions,
            vertex_colors: colors
                .iter()
                .map(|c| Rgba32Unmul::from_rgba_unmul_array(*c))
                .collect(),
            vertex_normals: normals,
            vertex_texcoords: uvs,
            materials: vec![material].into(),
        };

        Self {
            mesh_instances: vec![GpuMeshInstance {
                gpu_mesh: Arc::new(GpuMesh::new(re_ctx, &cpu_mesh).unwrap()),
                world_from_mesh: glam::Affine3A::from_translation(glam::vec3(0.0, 0.0, 0.0)),
                picking_layer_id: PickingLayerId::default(),
                additive_tint: Color32::WHITE,
                outline_mask_ids: Default::default(),
            }],
        }
    }

    fn draw(
        &mut self,
        re_ctx: &RenderContext,
        resolution: [u32; 2],
        time: &framework::Time,
        pixels_per_point: f32,
    ) -> anyhow::Result<Vec<framework::ViewDrawResult>> {
        Ok(vec![
            // 3D view
            {
                let secs_since_startup = time.secs_since_startup() / 5f32;
                let camera_rotation_center = glam::vec3(0f32, 0f32, 0f32);
                let camera_position =
                    glam::vec3(secs_since_startup.sin(), 0.5, secs_since_startup.cos()) * 10f32
                        + camera_rotation_center;
                let mut view_builder = ViewBuilder::new(
                    re_ctx,
                    view_builder::TargetConfiguration {
                        name: "3D".into(),
                        resolution_in_pixel: resolution,
                        view_from_world: re_math::IsoTransform::look_at_rh(
                            camera_position,
                            camera_rotation_center,
                            glam::Vec3::Y,
                        )
                        .unwrap(),
                        projection_from_view: Projection::Perspective {
                            vertical_fov: 70.0 * std::f32::consts::TAU / 360.0,
                            near_plane_distance: 0.01,
                            aspect_ratio: resolution[0] as f32 / resolution[1] as f32,
                        },
                        pixels_per_point,
                        ..Default::default()
                    },
                );

                view_builder.queue_draw(re_renderer::renderer::MeshDrawData::new(
                    re_ctx,
                    &self.mesh_instances,
                )?);
                let command_buffer = view_builder
                    .draw(re_ctx, re_renderer::Rgba::TRANSPARENT)
                    .unwrap();
                framework::ViewDrawResult {
                    view_builder,
                    command_buffer,
                    target_location: glam::Vec2::ZERO,
                }
            },
        ])
    }

    fn on_key_event(&mut self, _input: winit::event::KeyEvent) {}
}

fn main() {
    framework::start::<ExampleMesh>();
}

#[derive(Clone, Debug)]
pub struct MeshData {
    pub indices: Vec<UVec3>,
    pub positions: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub colors: Vec<[u8; 4]>,
    pub normals: Vec<Vec3>,
}

/// Builds a cube mesh data structure, we're using shared vertices for brevity.
fn build_cube_mesh_data() -> MeshData {
    let positions = vec![
        Vec3::new(-0.5, -0.5, 0.5),
        Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, 0.5),
        Vec3::new(-0.5, 0.5, 0.5),
        Vec3::new(-0.5, 0.5, -0.5),
        Vec3::new(0.5, 0.5, -0.5),
        Vec3::new(0.5, 0.5, 0.5),
    ];

    let indices = vec![
        UVec3::new(4, 5, 0),
        UVec3::new(5, 1, 0), // back
        UVec3::new(5, 6, 1),
        UVec3::new(6, 2, 1), // front
        UVec3::new(6, 7, 3),
        UVec3::new(2, 6, 3), // bottom
        UVec3::new(7, 4, 0),
        UVec3::new(3, 7, 0), // top
        UVec3::new(0, 1, 2),
        UVec3::new(3, 0, 2), // right
        UVec3::new(7, 6, 5),
        UVec3::new(4, 7, 5), // left
    ];

    let normals = positions.clone();
    let uvs = vec![Vec2::ZERO; positions.len()];
    let colors = vec![[200, 200, 200, 255]; positions.len()];

    MeshData {
        positions,
        indices,
        normals,
        uvs,
        colors,
    }
}
