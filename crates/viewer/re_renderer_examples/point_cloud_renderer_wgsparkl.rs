//! Point renderer for efficient rendering of point clouds.
//!
//!
//! How it works:
//! =================
//! Points are rendered as quads and stenciled out by a fragment shader.
//! Quad spanning happens in the vertex shader, uploaded are only the data for the actual points (no vertex buffer!).
//!
//! Like with the `super::lines::LineRenderer`, we're rendering as all quads in a single triangle list draw call.
//! (Rationale for this can be found in the [`lines.rs`]'s documentation)
//!
//! For WebGL compatibility, data is uploaded as textures. Color is stored in a separate srgb texture, meaning
//! that srgb->linear conversion happens on texture load.
//!

use re_renderer::{
    DrawPhase, GpuRenderPipelineHandle, GpuRenderPipelinePoolAccessor, RenderContext,
    RenderPipelineDesc, VertexBufferLayout,
    external::smallvec::smallvec,
    include_shader_module,
    renderer::{DrawData, DrawError, Renderer},
};

use crate::{InstanceData, graphics::InstanceBuffer, instancing3d::*};

/// A point cloud drawing operation.
/// Expected to be recreated every frame.
#[derive(Clone)]
pub struct PointCloudDrawData {
    prep_vertex_buffer: InstanceBuffer,
}

impl DrawData for PointCloudDrawData {
    type Renderer = PointCloudRenderer;
}

impl PointCloudDrawData {
    /// Transforms and uploads point cloud data to be consumed by gpu.
    ///
    /// Try to bundle all points into a single draw data instance whenever possible.
    /// Number of vertices and colors has to be equal.
    ///
    /// If no batches are passed, all points are assumed to be in a single batch with identity transform.
    pub fn new(prep_vertex_buffer: InstanceBuffer) -> Result<Self, ()> {
        Ok(Self { prep_vertex_buffer })
    }
}

pub struct PointCloudRenderer {
    pub render_pipeline: GpuRenderPipelineHandle,
}

impl Renderer for PointCloudRenderer {
    type RendererDrawData = PointCloudDrawData;

    fn participated_phases() -> &'static [DrawPhase] {
        &[DrawPhase::Opaque]
    }

    fn create_renderer(ctx: &RenderContext) -> Self {
        let instancing3d_shader_module = ctx
            .gpu_resources
            .shader_modules
            .get_or_create(ctx, &include_shader_module!("instancing3d.wgsl"));

        let surface_format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let pipeline_layout = get_pipeline_layout_rerun(&ctx);

        const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 9] = wgpu::vertex_attr_array![
            0 => Float32x3, // position
            1 => Float32x3, // normal
            2 => Float32x2, // uv

            3 => Float32x3, // def_x
            4 => Float32x3, // def_y
            5 => Float32x3, // def_z
            6 => Float32x3, // pos
            7 => Float32x4, // unused
            8 => Float32x4, // i_color
        ];
        let render_pipeline_desc = RenderPipelineDesc {
            label: "PointCloudPipeline".into(),
            pipeline_layout: pipeline_layout,
            vertex_entrypoint: "vertex".into(),
            vertex_handle: instancing3d_shader_module.clone(),
            fragment_entrypoint: "fragment".into(),
            fragment_handle: instancing3d_shader_module.clone(),
            vertex_buffers: smallvec![VertexBufferLayout {
                array_stride: std::mem::size_of::<InstanceData>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                // FIXME: we shouldnd't have to slice it: we're missing 1 item here!
                attributes: INSTANCE_ATTRIBUTES[..9].into(),
            }],
            // FIXME: Not sure about this one.
            render_targets: smallvec![Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },

            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        };
        let render_pipelines = &ctx.gpu_resources.render_pipelines;
        let render_pipeline = render_pipelines.get_or_create(ctx, &render_pipeline_desc);
        // TODO: create the pipeline thanks to code from `instancing3d.rs`
        Self { render_pipeline }
    }

    fn draw(
        &self,
        render_pipelines: &GpuRenderPipelinePoolAccessor<'_>,
        _phase: DrawPhase,
        pass: &mut wgpu::RenderPass<'_>,
        draw_data: &Self::RendererDrawData,
    ) -> Result<(), DrawError> {
        // TODO: store render pipeline in self.

        let render_pipeline = render_pipelines.get(self.render_pipeline)?;
        pass.set_pipeline(&render_pipeline);
        pass.set_vertex_buffer(1, draw_data.prep_vertex_buffer.buffer.slice(..));
        // FIXME: only 1 point cloud instance allowed here.
        pass.draw(0..draw_data.prep_vertex_buffer.length as u32, 0..1);

        Ok(())
    }
}
