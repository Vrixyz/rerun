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
    RenderContext,
    renderer::{DrawData, Renderer},
};

use crate::instancing3d::create_render_pipeline;

/// A point cloud drawing operation.
/// Expected to be recreated every frame.
#[derive(Clone)]
pub struct PointCloudDrawData {
    prep_vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
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
    pub fn new(
        ctx: &RenderContext,
        prep_vertex_buffer: wgpu::Buffer,
        vertex_count: u32,
    ) -> Result<Self, ()> {
        let point_renderer = ctx.renderer::<PointCloudRenderer>();
        Ok(Self {
            prep_vertex_buffer,
            vertex_count,
        })
    }
}

pub struct PointCloudRenderer {}

impl Renderer for PointCloudRenderer {
    type RendererDrawData = PointCloudDrawData;

    fn participated_phases() -> &'static [DrawPhase] {
        &[DrawPhase::Opaque]
    }

    fn create_renderer(ctx: &RenderContext) -> Self {
        let render_pipeline = create_render_pipeline(&ctx.device);
        // TODO: create the pipeline thanks to code from `instancing3d.rs`
        Self { render_pipeline }
    }

    fn draw(
        &self,
        render_pipelines: &GpuRenderPipelinePoolAccessor<'_>,
        phase: DrawPhase,
        pass: &mut wgpu::RenderPass<'_>,
        draw_data: &Self::RendererDrawData,
    ) -> Result<(), DrawError> {
        // TODO: store render pipeline in self.

        pass.set_pipeline(&self.render_pipeline);
        pass.set_vertex_buffer(1, draw_data.prep_vertex_buffer.slice(..));
        // FIXME: only 1 point cloud instance allowed here.
        pass.draw(0..self.vertex_count, 0..1);

        Ok(())
    }
}
