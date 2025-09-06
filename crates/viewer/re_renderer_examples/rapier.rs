//! Example displaying rapier ; inspired from other examples.

// TODO(#6330): remove unwrap()
#![allow(clippy::unwrap_used)]

use nalgebra::Vector3;
use re_renderer::{
    Color32, LineBatchBuilder, LineDrawableBuilder, Size,
    renderer::LineStripFlags,
    view_builder::{self, Projection, ViewBuilder},
};

use rapier3d::pipeline::{
    DebugColor, DebugRenderBackend, DebugRenderMode, DebugRenderObject, DebugRenderPipeline,
};
use rapier3d::prelude::*;

mod framework;

struct RenderRapier {
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    pub impulse_joints: ImpulseJointSet,
    pub multibody_joints: MultibodyJointSet,
    pub physics_pipeline: PhysicsPipeline,
    pub ccd_solver: CCDSolver,
    pub narrow_phase: NarrowPhase,
    pub broad_phase: DefaultBroadPhase,
    pub island_manager: IslandManager,

    pub render_pipeline: DebugRenderPipeline,
}

impl framework::Example for RenderRapier {
    fn title() -> &'static str {
        "Rapier 3D"
    }

    fn new(re_ctx: &re_renderer::RenderContext) -> Self {
        // From rapier's debug_boxes3 example.
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let impulse_joints = ImpulseJointSet::new();
        let multibody_joints = MultibodyJointSet::new();
        let narrow_phase = NarrowPhase::new();

        let ground_size = 100.1;
        let ground_height = 0.1;

        /*
         * Ground
         */
        let ground_size = 50.0;
        let ground_height = 0.1;

        let rigid_body = RigidBodyBuilder::fixed().translation(vector![0.0, -ground_height, 0.0]);
        let ground_handle = bodies.insert(rigid_body);
        let collider = ColliderBuilder::cuboid(ground_size, ground_height, ground_size);
        colliders.insert_with_parent(collider, ground_handle, &mut bodies);

        fn create_pyramid(
            bodies: &mut RigidBodySet,
            colliders: &mut ColliderSet,
            offset: Vector<f32>,
            stack_height: usize,
            half_extents: Vector<f32>,
        ) {
            let shift = half_extents * 2.5;
            for i in 0usize..stack_height {
                for j in i..stack_height {
                    for k in i..stack_height {
                        let fi = i as f32;
                        let fj = j as f32;
                        let fk = k as f32;
                        let x = (fi * shift.x / 2.0) + (fk - fi) * shift.x + offset.x
                            - stack_height as f32 * half_extents.x;
                        let y = fi * shift.y + offset.y;
                        let z = (fi * shift.z / 2.0) + (fj - fi) * shift.z + offset.z
                            - stack_height as f32 * half_extents.z;

                        // Build the rigid body.
                        let rigid_body = RigidBodyBuilder::dynamic().translation(vector![x, y, z]);
                        let rigid_body_handle = bodies.insert(rigid_body);

                        let collider =
                            ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z);
                        colliders.insert_with_parent(collider, rigid_body_handle, bodies);
                    }
                }
            }
        }
        /*
         * Create the cubes
         */
        let cube_size = 1.0;
        let hext = Vector::repeat(cube_size);
        let bottomy = cube_size;
        create_pyramid(
            &mut bodies,
            &mut colliders,
            vector![0.0, bottomy, 0.0],
            18,
            hext,
        );

        Self {
            bodies,
            colliders,
            impulse_joints,
            multibody_joints,
            physics_pipeline: PhysicsPipeline::new(),
            ccd_solver: CCDSolver::new(),
            narrow_phase,
            broad_phase: DefaultBroadPhase::new(),
            island_manager: IslandManager::new(),
            render_pipeline: DebugRenderPipeline::default(),
        }
    }

    fn draw(
        &mut self,
        re_ctx: &re_renderer::RenderContext,
        resolution: [u32; 2],
        time: &framework::Time,
        pixels_per_point: f32,
    ) -> anyhow::Result<Vec<framework::ViewDrawResult>> {
        let screen_size = glam::vec2(resolution[0] as f32, resolution[1] as f32);

        let mut line_strip_builder = LineDrawableBuilder::new(re_ctx);
        line_strip_builder.reserve_strips(12800).unwrap();
        line_strip_builder.reserve_vertices(204800).unwrap();

        self.physics_pipeline.step(
            &Vector3::new(0.0, -9.81, 0.0),
            &IntegrationParameters::default(),
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            None, //Some(&mut self.query_pipeline),
            &(),  //&*physics.hooks,
            &(),
        );
        // debug render pipeline
        {
            let mut line_batch = line_strip_builder.batch("lines");

            let mut pipeline = RerunRenderPipeline {
                line_batch_builder: line_batch,
            };

            self.render_pipeline.render(
                &mut pipeline,
                &self.bodies,
                &self.colliders,
                &self.impulse_joints,
                &self.multibody_joints,
                &self.narrow_phase,
            );
        }

        let line_strip_draw_data = line_strip_builder.into_draw_data()?;

        Ok(vec![
            // 3D view
            {
                let secs_since_startup = time.secs_since_startup();
                let camera_rotation_center = glam::vec3(0f32, 20f32, 0f32);
                let camera_position =
                    glam::vec3(secs_since_startup.sin(), 0.5, secs_since_startup.cos()) * 50f32
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
                let command_buffer = view_builder
                    .queue_draw(line_strip_draw_data)
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
    framework::start::<RenderRapier>();
}

pub struct RerunRenderPipeline<'a, 'b> {
    pub line_batch_builder: LineBatchBuilder<'a, 'b>,
}

impl<'a, 'b> DebugRenderBackend for RerunRenderPipeline<'a, 'b> {
    fn draw_line(
        &mut self,
        object: DebugRenderObject,
        a: Point<Real>,
        b: Point<Real>,
        color: DebugColor,
    ) {
        self.line_batch_builder
            .add_segment(a.into(), b.into())
            .radius(Size::new_scene_units(0.05f32))
            .color(Color32::from_rgba_unmultiplied(
                (color[0] * 255.0) as u8,
                (color[1] * 255.0) as u8,
                (color[2] * 255.0) as u8,
                (color[3] * 255.0) as u8,
            ))
            .flags(LineStripFlags::empty() | LineStripFlags::FLAG_COLOR_GRADIENT);
    }
}
