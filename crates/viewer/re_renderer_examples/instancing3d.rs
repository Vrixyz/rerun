use re_renderer::{
    BindGroupLayoutDesc, GpuPipelineLayoutHandle, PipelineLayoutDesc, RenderContext,
};
use wgpu::{PipelineLayout, RenderPipeline, VertexBufferLayout};

use crate::InstanceData;

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

pub fn get_layout() -> VertexBufferLayout<'static> {
    VertexBufferLayout {
        array_stride: std::mem::size_of::<InstanceData>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &INSTANCE_ATTRIBUTES,
    }
}

pub fn load_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Instancing3d"),
        source: wgpu::ShaderSource::Wgsl(include_str!("instancing3d.wgsl").into()),
    })
}

pub fn get_pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
    let camera_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("camera_bind_group_layout"),
        });
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("PointCloudPipelineLayout"),
        bind_group_layouts: &[&camera_bind_group_layout],
        push_constant_ranges: &[],
    })
}
pub fn get_pipeline_layout_rerun(ctx: &RenderContext) -> GpuPipelineLayoutHandle {
    let bind_group_layout = ctx.gpu_resources.bind_group_layouts.get_or_create(
        &ctx.device,
        &BindGroupLayoutDesc {
            label: "DepthCopyWorkaround::render_pipeline".into(),
            entries: vec![wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        },
    );

    ctx.gpu_resources.pipeline_layouts.get_or_create(
        ctx,
        &PipelineLayoutDesc {
            label: "DepthCopyWorkaround::render_pipeline".into(),
            entries: vec![ctx.global_bindings.layout, bind_group_layout],
        },
    );
    let vertex_bind_group_layout = ctx.gpu_resources.bind_group_layouts.get_or_create(
        &ctx.device,
        &BindGroupLayoutDesc {
            label: "vertex_bind_group_layout".into(),
            entries: vec![wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        },
    );

    ctx.gpu_resources.pipeline_layouts.get_or_create(
        ctx,
        &PipelineLayoutDesc {
            label: "PointCloudPipelineLayout".into(),
            entries: vec![ctx.global_bindings.layout, vertex_bind_group_layout],
        },
    )
}

pub fn create_render_pipeline(device: &wgpu::Device) -> RenderPipeline {
    let shader_module = load_shader_module(device);
    let surface_format = wgpu::TextureFormat::Bgra8UnormSrgb;
    let pipeline_layout = get_pipeline_layout(device);
    let instance_buffer_layout = get_layout();
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("PointCloudPipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader_module,
            entry_point: Some("vertex"),
            buffers: &[instance_buffer_layout],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader_module,
            entry_point: Some("fragment"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
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
        multiview: None,
        cache: None,
    })
}
