//! Example displaying rapier ; inspired from other examples.

// TODO(#6330): remove unwrap()
#![allow(clippy::unwrap_used)]

use core::num;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::Vec4;
use nalgebra::Vector3;
use re_renderer::{
    Color32, LineBatchBuilder, LineDrawableBuilder, PointCloudBuilder, RenderContext, Size,
    renderer::{LineStripFlags, PointCloudDrawData},
    view_builder::{self, Projection, ViewBuilder},
};

use rapier3d::pipeline::{DebugColor, DebugRenderBackend, DebugRenderObject, DebugRenderPipeline};
use rapier3d::prelude::*;

use wgpu::Buffer;

use wgcore::re_exports::encase::StorageBuffer;
use wgsparkl3d::{
    models::{DruckerPrager, ElasticCoefficients},
    pipeline::{MpmData, MpmPipeline},
    rapier::prelude::RigidBodyPosition,
    solver::{Particle, ParticleDynamics, SimulationParams},
    wgparry::math::GpuSim,
    wgrapier::dynamics::GpuVelocity,
};

mod framework;

#[derive(Default)]
pub struct RapierData {
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    pub impulse_joints: ImpulseJointSet,
    pub multibody_joints: MultibodyJointSet,
    pub integration_parameters: IntegrationParameters,
    pub physics_pipeline: PhysicsPipeline,
    pub narrow_phase: NarrowPhase,
    pub broad_phase: DefaultBroadPhase,
    pub ccd_solver: CCDSolver,
    pub islands: IslandManager,
}

struct RenderWgSparkl {
    pub rapier_data: RapierData,
    pub render_pipeline: DebugRenderPipeline,
    pub mpm_data: MpmData,
    pub mpm_pipeline: MpmPipeline,
    pub num_substeps: usize,
}

#[derive(Clone)]
pub struct InstanceMaterialData {
    pub data: Vec<InstanceData>,
    pub buffer: InstanceBuffer,
}
#[derive(Clone)]
pub struct InstanceBuffer {
    pub buffer: Arc<Buffer>,
    pub length: usize,
}

#[derive(Clone, Copy, Pod, Zeroable, Default)]
#[repr(C)]
pub struct InstanceData {
    pub deformation: [Vec4; 3],
    pub position: Vec4,
    pub base_color: [f32; 4],
    pub color: [f32; 4],
}

impl framework::Example for RenderWgSparkl {
    fn title() -> &'static str {
        "Rapier 3D"
    }

    fn new(re_ctx: &re_renderer::RenderContext) -> Self {
        let pipeline = MpmPipeline::new(&re_ctx.device).unwrap();

        let mut rapier_data = RapierData {
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            narrow_phase: NarrowPhase::new(),
            broad_phase: DefaultBroadPhase::new(),
            ccd_solver: CCDSolver::new(),
            islands: IslandManager::new(),
        };

        /*
         * Ground
         */
        let ground_size = 50.0;
        let ground_height = 2.0;

        let rigid_body = RigidBodyBuilder::fixed().translation(vector![0.0, -ground_height, 0.0]);
        let ground_handle = rapier_data.bodies.insert(rigid_body);
        let collider = ColliderBuilder::cuboid(ground_size, ground_height, ground_size);
        rapier_data
            .colliders
            .insert_with_parent(collider, ground_handle, &mut rapier_data.bodies);

        let nxz = 45;
        let cell_width = 1.0;
        let mut particles = vec![];
        for i in 0..nxz {
            for j in 0..100 {
                for k in 0..nxz {
                    let position = vector![
                        i as f32 + 0.5 - nxz as f32 / 2.0,
                        j as f32 + 0.5 + 10.0,
                        k as f32 + 0.5 - nxz as f32 / 2.0
                    ] * cell_width
                        / 2.0;
                    let density = 2700.0;
                    let radius = cell_width / 4.0;
                    particles.push(Particle {
                        position,
                        dynamics: ParticleDynamics::with_density(radius, density),
                        model: ElasticCoefficients::from_young_modulus(2_000_000_000.0, 0.2),
                        plasticity: Some(DruckerPrager::new(2_000_000_000.0, 0.2)),
                        phase: None,
                    });
                }
            }
        }
        let num_substeps = 20;
        let mpm_data = MpmData::new(
            &re_ctx.device,
            SimulationParams {
                dt: (1.0 / 60.0) / num_substeps as f32,
                gravity: Vector3::new(0.0, -9.81, 0.0),
            },
            &particles,
            &rapier_data.bodies,
            &rapier_data.colliders,
            cell_width,
            60_000,
        );
        Self {
            rapier_data,
            render_pipeline: DebugRenderPipeline::default(),
            mpm_data,
            mpm_pipeline: pipeline,
            num_substeps,
        }
    }

    fn draw(
        &mut self,
        re_ctx: &re_renderer::RenderContext,
        resolution: [u32; 2],
        time: &framework::Time,
        pixels_per_point: f32,
    ) -> anyhow::Result<Vec<framework::ViewDrawResult>> {
        let mut line_strip_builder = LineDrawableBuilder::new(re_ctx);
        line_strip_builder.reserve_strips(12800).unwrap();
        line_strip_builder.reserve_vertices(204800).unwrap();

        let point_draw_data = run_simulation(re_ctx, self);
        // debug render pipeline
        {
            let line_batch = line_strip_builder.batch("lines");

            let mut pipeline = RerunRenderPipeline {
                line_batch_builder: line_batch,
            };

            self.render_pipeline.render(
                &mut pipeline,
                &self.rapier_data.bodies,
                &self.rapier_data.colliders,
                &self.rapier_data.impulse_joints,
                &self.rapier_data.multibody_joints,
                &self.rapier_data.narrow_phase,
            );
        }

        let line_strip_draw_data = line_strip_builder.into_draw_data()?;

        Ok(vec![
            // 3D view
            {
                let secs_since_startup = time.secs_since_startup() / 50f32;
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
                view_builder.queue_draw(point_draw_data.clone());
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
    framework::start::<RenderWgSparkl>();
}

pub struct RerunRenderPipeline<'a, 'b> {
    pub line_batch_builder: LineBatchBuilder<'a, 'b>,
}

impl<'a, 'b> DebugRenderBackend for RerunRenderPipeline<'a, 'b> {
    fn draw_line(
        &mut self,
        _object: DebugRenderObject<'_>,
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

/// Inspired from wgsparkl testbed::step
fn run_simulation(ctx: &RenderContext, physics: &mut RenderWgSparkl) -> PointCloudDrawData {
    // Run the simulation.
    let mut encoder = ctx.device.create_command_encoder(&Default::default());

    // Send updated bodies information to the gpu.
    // PERF: don’t reallocate the buffers at each step.
    let poses_data: Vec<GpuSim> = physics
        .mpm_data
        .coupling()
        .iter()
        .map(|coupling| {
            let c = &physics.rapier_data.colliders[coupling.collider];
            //#[cfg(feature = "dim2")]
            //return (*c.position()).into();
            //#[cfg(feature = "dim3")]
            return GpuSim::from_isometry(*c.position(), 1.0);
        })
        .collect();
    // println!("poses: {:?}", poses_data);
    ctx.queue.write_buffer(
        physics.mpm_data.bodies.poses().buffer(),
        0,
        bytemuck::cast_slice(&poses_data),
    );

    //// Copy the velocities to the GPU.

    let divisor = physics.num_substeps as f32; // app_state.num_substeps as f32;
    let gravity = Vector::y() * -9.81;
    let vels_data: Vec<_> = physics
        .mpm_data
        .coupling()
        .iter()
        .map(|coupling| {
            let rb = &physics.rapier_data.bodies[coupling.body];
            GpuVelocity {
                linear: *rb.linvel()
                    + gravity
                        * physics.rapier_data.integration_parameters.dt
                        * (rb.is_dynamic() as u32 as f32)
                        / divisor,
                #[allow(clippy::clone_on_copy)] // Needed for the 2d/3d switch.
                angular: rb.angvel().clone()* physics.rapier_data.integration_parameters.dt / divisor,
            }
        })
        .collect();

    let mut vels_bytes = vec![];
    let mut buffer = StorageBuffer::new(&mut vels_bytes);
    buffer.write(&vels_data).unwrap();
    ctx.queue
        .write_buffer(physics.mpm_data.bodies.vels().buffer(), 0, &vels_bytes);

    //// Step the simulation.
    for _ in 0..physics.num_substeps {
        physics
            .mpm_pipeline
            .dispatch_step(&ctx.device, &mut encoder, &mut physics.mpm_data, None);
    }
    physics
        .mpm_data
        .poses_staging
        .copy_from(&mut encoder, physics.mpm_data.bodies.poses());

    // TODO: timings
    // TODO: rendering vertex buffer preparation
    // - ideally this should be made in GPU be reading the same particle buffer.
    // - A "simpler" (more easily debuggable) solution is to read the particle buffer (from MpmData::GpuPraticles), then dispatch a cloud point render.
    //    - This will enable to test current logic before implementing more complex shaders.

    ctx.queue.submit(Some(encoder.finish()));

    // Read to CPU implementation
    // TODO: move that to GPU
    let points = read_particles::read_particles_positions(
        &ctx.device,
        &ctx.queue,
        &physics.mpm_data.particles,
    )
    .iter()
    .map(|p| glam::vec3(p.x, p.y, p.z))
    .collect::<Vec<_>>();
    //dbg!(&points);
    let mut point_cloud_builder = PointCloudBuilder::new(ctx);
    point_cloud_builder
        .batch("mpm particles point cloud")
        .add_points(
            &points,
            &points
                .iter()
                .map(|_| Size::new_scene_units(0.2f32))
                .collect::<Vec<_>>(),
            &points
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    Color32::from_rgb(
                        (i * 5 % 255) as u8,
                        (i * 7 % 255) as u8,
                        (i * 11 % 255) as u8,
                    )
                })
                .collect::<Vec<_>>(),
            &[],
        );
    let point_draw_data = point_cloud_builder.into_draw_data().unwrap();

    // end read to CPU + display.

    // FIXME: make the readback work on wasm too.
    //        Currently, this means there won’t be any two-ways coupling on wasm.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let new_poses =
            futures::executor::block_on(physics.mpm_data.poses_staging.read(&ctx.device)).unwrap();

        // println!("Impulses: {:?}", new_poses);

        for (i, coupling) in physics.mpm_data.coupling().iter().enumerate() {
            let rb = &mut physics.rapier_data.bodies[coupling.body];
            if rb.is_dynamic() {
                let interpolator = RigidBodyPosition {
                    position: *rb.position(),
                    //#[cfg(feature = "dim2")]
                    //next_position: new_poses[i].similarity.isometry,
                    //#[cfg(feature = "dim3")]
                    next_position: new_poses[i].isometry,
                };
                let vel = interpolator.interpolate_velocity(
                    1.0 / (physics.rapier_data.integration_parameters.dt / divisor),
                    &rb.mass_properties().local_mprops.local_com,
                );
                rb.set_linvel(vel.linvel, true);
                rb.set_angvel(vel.angvel, true);
                // println!("dvel: {:?}", vel.linvel - vel_before);
            }
        }
    }
    let mut params = physics.rapier_data.integration_parameters;
    params.dt /= divisor;
    physics.rapier_data.physics_pipeline.step(
        &nalgebra::zero(),
        &params, // physics.rapier_data.params,
        &mut physics.rapier_data.islands,
        &mut physics.rapier_data.broad_phase,
        &mut physics.rapier_data.narrow_phase,
        &mut physics.rapier_data.bodies,
        &mut physics.rapier_data.colliders,
        &mut physics.rapier_data.impulse_joints,
        &mut physics.rapier_data.multibody_joints,
        &mut physics.rapier_data.ccd_solver,
        None,
        &(),
        &(),
    );
    point_draw_data
}

pub mod read_particles {
    use super::*;

    use nalgebra::Vector4;
    use wgcore::tensor::GpuVector;
    use wgsparkl3d::solver::GpuParticles;

    pub fn read_particles_positions(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particles: &GpuParticles,
    ) -> Vec<Vector4<f32>> {
        // Create the staging buffer.
        // Here `particles` is of type `GpuParticles`, accessible from
        // `PhysicsContext::data::particles`.
        let positions_staging: GpuVector<Vector4<f32>> = GpuVector::uninit(
            device,
            particles.len() as u32,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );

        // Copy the buffer.
        let mut encoder = device.create_command_encoder(&Default::default());
        positions_staging.copy_from(&mut encoder, &particles.positions);
        queue.submit(Some(encoder.finish()));

        // Run the copy. The fourth component of each entry can be ignored.
        let positions: Vec<Vector4<f32>> =
            futures::executor::block_on(positions_staging.read(device)).unwrap();
        positions
    }
}
