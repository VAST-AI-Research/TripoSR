import numpy as np
import torch
import xatlas
import trimesh
import moderngl
from PIL import Image


def make_atlas(mesh, texture_resolution, texture_padding):
    atlas = xatlas.Atlas()
    atlas.add_mesh(mesh.vertices, mesh.faces)
    options = xatlas.PackOptions()
    options.resolution = texture_resolution
    options.padding = texture_padding
    options.bilinear = True
    atlas.generate(pack_options=options)
    vmapping, indices, uvs = atlas[0]
    return {
        "vmapping": vmapping,
        "indices": indices,
        "uvs": uvs,
    }


def rasterize_position_atlas(
    mesh, atlas_vmapping, atlas_indices, atlas_uvs, texture_resolution, texture_padding
):
    ctx = moderngl.create_context(standalone=True)
    basic_prog = ctx.program(
        vertex_shader="""
            #version 330
            in vec2 in_uv;
            in vec3 in_pos;
            out vec3 v_pos;
            void main() {
                v_pos = in_pos;
                gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
            }
        """,
        fragment_shader="""
            #version 330
            in vec3 v_pos;
            out vec4 o_col;
            void main() {
                o_col = vec4(v_pos, 1.0);
            }
        """,
    )
    gs_prog = ctx.program(
        vertex_shader="""
            #version 330
            in vec2 in_uv;
            in vec3 in_pos;
            out vec3 vg_pos;
            void main() {
                vg_pos = in_pos;
                gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
            }
        """,
        geometry_shader="""
            #version 330
            uniform float u_resolution;
            uniform float u_dilation;
            layout (triangles) in;
            layout (triangle_strip, max_vertices = 12) out;
            in vec3 vg_pos[];
            out vec3 vf_pos;
            void lineSegment(int aidx, int bidx) {
                vec2 a = gl_in[aidx].gl_Position.xy;
                vec2 b = gl_in[bidx].gl_Position.xy;
                vec3 aCol = vg_pos[aidx];
                vec3 bCol = vg_pos[bidx];

                vec2 dir = normalize((b - a) * u_resolution);
                vec2 offset = vec2(-dir.y, dir.x) * u_dilation / u_resolution;

                gl_Position = vec4(a + offset, 0.0, 1.0);
                vf_pos = aCol;
                EmitVertex();
                gl_Position = vec4(a - offset, 0.0, 1.0);
                vf_pos = aCol;
                EmitVertex();
                gl_Position = vec4(b + offset, 0.0, 1.0);
                vf_pos = bCol;
                EmitVertex();
                gl_Position = vec4(b - offset, 0.0, 1.0);
                vf_pos = bCol;
                EmitVertex();
            }
            void main() {
                lineSegment(0, 1);
                lineSegment(1, 2);
                lineSegment(2, 0);
                EndPrimitive();
            }
        """,
        fragment_shader="""
            #version 330
            in vec3 vf_pos;
            out vec4 o_col;
            void main() {
                o_col = vec4(vf_pos, 1.0);
            }
        """,
    )
    uvs = atlas_uvs.flatten().astype("f4")
    pos = mesh.vertices[atlas_vmapping].flatten().astype("f4")
    indices = atlas_indices.flatten().astype("i4")
    vbo_uvs = ctx.buffer(uvs)
    vbo_pos = ctx.buffer(pos)
    ibo = ctx.buffer(indices)
    vao_content = [
        vbo_uvs.bind("in_uv", layout="2f"),
        vbo_pos.bind("in_pos", layout="3f"),
    ]
    basic_vao = ctx.vertex_array(basic_prog, vao_content, ibo)
    gs_vao = ctx.vertex_array(gs_prog, vao_content, ibo)
    fbo = ctx.framebuffer(
        color_attachments=[
            ctx.texture((texture_resolution, texture_resolution), 4, dtype="f4")
        ]
    )
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 0.0)
    gs_prog["u_resolution"].value = texture_resolution
    gs_prog["u_dilation"].value = texture_padding
    gs_vao.render()
    basic_vao.render()

    fbo_bytes = fbo.color_attachments[0].read()
    fbo_np = np.frombuffer(fbo_bytes, dtype="f4").reshape(
        texture_resolution, texture_resolution, 4
    )
    return fbo_np


def positions_to_colors(model, scene_code, positions_texture, texture_resolution):
    positions = torch.tensor(positions_texture.reshape(-1, 4)[:, :-1])
    with torch.no_grad():
        queried_grid = model.renderer.query_triplane(
            model.decoder,
            positions,
            scene_code,
        )
    rgb_f = queried_grid["color"].numpy().reshape(-1, 3)
    rgba_f = np.insert(rgb_f, 3, positions_texture.reshape(-1, 4)[:, -1], axis=1)
    rgba_f[rgba_f[:, -1] == 0.0] = [0, 0, 0, 0]
    return rgba_f.reshape(texture_resolution, texture_resolution, 4)


def bake_texture(mesh, model, scene_code, texture_resolution):
    texture_padding = round(max(2, texture_resolution / 256))
    atlas = make_atlas(mesh, texture_resolution, texture_padding)
    positions_texture = rasterize_position_atlas(
        mesh,
        atlas["vmapping"],
        atlas["indices"],
        atlas["uvs"],
        texture_resolution,
        texture_padding,
    )
    colors_texture = positions_to_colors(
        model, scene_code, positions_texture, texture_resolution
    )
    return {
        "vmapping": atlas["vmapping"],
        "indices": atlas["indices"],
        "uvs": atlas["uvs"],
        "colors": colors_texture,
    }
