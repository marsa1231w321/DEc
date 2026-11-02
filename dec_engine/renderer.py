from itertools import chain

import moderngl
import numpy as np
from PIL import Image


class Renderer:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.shadow_size = 2048
        self._shadow_signature = None
        self._light_space_matrix = None
        self._mesh_cache = {}
        self._rig_cache = {}
        self._texture_cache = {}
        self._output_target = None

        self.shadow_program = ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 light_space_matrix;
                uniform mat4 model;
                in vec3 in_position;
                void main() {
                    gl_Position = light_space_matrix * model * vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                void main() { }
            """,
        )

        self.gizmo_program = ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 projection;
                uniform mat4 view;
                uniform mat4 model;
                in vec3 in_position;
                void main() {
                    gl_Position = projection * view * model * vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                uniform vec3 color;
                out vec4 f_color;
                void main() { f_color = vec4(color, 1.0); }
            """,
        )
        self.gizmos = {}
        self._create_gizmo("light", self._diamond_lines())
        self._create_gizmo("light_direction", self._arrow_lines())
        self._create_gizmo("spawn", self._spawn_lines())
        self._create_gizmo("sound", self._sound_lines())
        self._create_gizmo("range", self._range_lines())
        self._create_gizmo("technical", self._technical_lines())
        self.rig_buffer = ctx.buffer(reserve=16384 * 3 * 4, dynamic=True)
        self.rig_vao = ctx.vertex_array(
            self.gizmo_program, [(self.rig_buffer, "3f", "in_position")]
        )

        self.program = ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 projection;
                uniform mat4 view;
                uniform mat4 model;
                uniform mat4 light_space_matrix;
                uniform vec2 texture_repeat;

                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_uv;

                out vec3 v_world_position;
                out vec3 v_normal;
                out vec4 v_shadow_coord;
                out vec2 v_uv;

                void main() {
                    vec4 world = model * vec4(in_position, 1.0);
                    v_world_position = world.xyz;
                    v_normal = transpose(inverse(mat3(model))) * in_normal;
                    v_shadow_coord = light_space_matrix * world;
                    v_uv = in_uv * texture_repeat;
                    gl_Position = projection * view * world;
                }
            """,
            fragment_shader="""
                #version 330
                uniform vec3 u_camera_pos;
                uniform vec3 light_dir;
                uniform vec3 light_color;
                uniform vec3 base_color;
                uniform float light_intensity;
                uniform float ambient_intensity;
                uniform int shadows_enabled;
                uniform int point_light_count;
                uniform vec3 point_light_positions[4];
                uniform vec3 point_light_colors[4];
                uniform float point_light_intensities[4];
                uniform sampler2D shadow_map;
                uniform sampler2D surface_texture;
                uniform int has_texture;

                in vec3 v_world_position;
                in vec3 v_normal;
                in vec4 v_shadow_coord;
                in vec2 v_uv;

                out vec4 f_color;

                float sample_shadow(vec4 shadow_coord) {
                    if (shadows_enabled == 0) return 1.0;
                    vec3 proj = shadow_coord.xyz / shadow_coord.w;
                    proj = proj * 0.5 + 0.5;
                    if (any(lessThan(proj, vec3(0.0))) || any(greaterThan(proj, vec3(1.0)))) return 1.0;
                    float bias = max(0.02 * (1.0 - dot(normalize(v_normal), normalize(light_dir))), 0.004);
                    vec2 texel = 1.0 / vec2(textureSize(shadow_map, 0));
                    float visibility = 0.0;
                    for (int x = -1; x <= 1; ++x) {
                        for (int y = -1; y <= 1; ++y) {
                            float closest = texture(shadow_map, proj.xy + vec2(x, y) * texel).r;
                            visibility += proj.z - bias > closest ? 0.4 : 1.0;
                        }
                    }
                    return visibility / 9.0;
                }

                void main() {
                    vec3 N = normalize(v_normal);
                    vec3 L = normalize(light_dir);
                    float diff = max(dot(N, -L), 0.0);

                    vec4 texel = has_texture == 1 ? texture(surface_texture, v_uv) : vec4(1.0);
                    vec3 albedo = has_texture == 1 ? texel.rgb : base_color;
                    vec3 ambient = albedo * ambient_intensity;
                    vec3 diffuse = albedo * (0.12 + diff * light_intensity) * light_color;
                    vec3 view_dir = normalize(u_camera_pos - v_world_position);
                    vec3 reflect_dir = reflect(-L, N);
                    float spec = pow(max(dot(reflect_dir, view_dir), 0.0), 28.0) * light_intensity;

                    float shadow = sample_shadow(v_shadow_coord);
                    vec3 shaded = (ambient + diffuse * shadow + spec * 0.3 * light_color * shadow);
                    for (int i = 0; i < point_light_count; ++i) {
                        vec3 to_light = point_light_positions[i] - v_world_position;
                        float distance_to_light = length(to_light);
                        vec3 point_dir = to_light / max(distance_to_light, 0.001);
                        float attenuation = point_light_intensities[i] /
                            (1.0 + 0.08 * distance_to_light + 0.02 * distance_to_light * distance_to_light);
                        float point_diffuse = max(dot(N, point_dir), 0.0);
                        vec3 point_reflection = reflect(-point_dir, N);
                        float point_specular = pow(max(dot(point_reflection, view_dir), 0.0), 28.0);
                        shaded += albedo * point_light_colors[i] * point_diffuse * attenuation;
                        shaded += point_light_colors[i] * point_specular * attenuation * 0.2;
                    }
                    f_color = vec4(shaded, texel.a);
                }
            """,
        )

        self.shadow_tex = ctx.depth_texture((self.shadow_size, self.shadow_size))
        self.shadow_fbo = ctx.framebuffer(depth_attachment=self.shadow_tex)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.front_face = 'ccw'

    def build_mesh(self, mesh, program=None, vertices=None):
        vertices = mesh.vertices if vertices is None else vertices
        cached = self._mesh_cache.get(id(mesh))
        if cached is not None and (cached[0] is not mesh or cached[1] != id(vertices)):
            _, _, main_vao, shadow_vao, vbo, ibo = cached
            main_vao.release()
            shadow_vao.release()
            vbo.release()
            ibo.release()
            cached = None
        if cached is None:
            attribs = np.hstack([vertices, mesh.normals, mesh.uvs]).astype(np.float32)
            vbo = self.ctx.buffer(attribs.tobytes())
            ibo = self.ctx.buffer(mesh.indices.astype(np.int32).tobytes())
            main_vao = self.ctx.vertex_array(
                self.program,
                [(vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_uv')],
                index_buffer=ibo,
            )
            shadow_vao = self.ctx.vertex_array(
                self.shadow_program,
                [(vbo, '3f 20x', 'in_position')],
                index_buffer=ibo,
            )
            cached = (mesh, id(vertices), main_vao, shadow_vao, vbo, ibo)
            self._mesh_cache[id(mesh)] = cached
        if program is self.shadow_program:
            return cached[3]
        return cached[2]

    def _deformed_vertices(self, mesh):
        rig = mesh.rig
        if not rig:
            return mesh.vertices
        bones = rig["bones"]
        signature = (
            id(rig["base_vertices"]),
            id(rig["weights"]),
            tuple(tuple(float(value) for value in bone["rotation"]) for bone in bones),
        )
        cached = self._rig_cache.get(id(mesh))
        if cached is not None and cached[0] == signature:
            return cached[1]
        base = np.asarray(rig["base_vertices"], dtype=np.float32)
        weights = np.asarray(rig["weights"], dtype=np.float32)
        deformed = base.copy()
        for index, bone in enumerate(bones):
            x, y, z = (float(value) for value in bone["rotation"])
            cx, sx = np.cos(x), np.sin(x)
            cy, sy = np.cos(y), np.sin(y)
            cz, sz = np.cos(z), np.sin(z)
            rx = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=np.float32)
            ry = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float32)
            rz = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=np.float32)
            pivot = np.asarray(bone["head"], dtype=np.float32)
            moved = (base - pivot) @ (rz @ ry @ rx).T + pivot
            deformed += (moved - base) * weights[:, index, None]
        self._rig_cache[id(mesh)] = (signature, deformed)
        return deformed

    @staticmethod
    def _bone_tail(bone):
        x, y, z = (float(value) for value in bone["rotation"])
        cx, sx = np.cos(x), np.sin(x)
        cy, sy = np.cos(y), np.sin(y)
        cz, sz = np.cos(z), np.sin(z)
        rx = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=np.float32)
        ry = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float32)
        rz = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=np.float32)
        head = np.asarray(bone["head"], dtype=np.float32)
        tail = np.asarray(bone["tail"], dtype=np.float32)
        return head + (rz @ ry @ rx) @ (tail - head)

    def _release_unused_meshes(self, scene, preview_object=None):
        active = {id(obj) for obj in scene.objects}
        if preview_object is not None:
            active.add(id(preview_object))
        for mesh_id in list(self._mesh_cache):
            if mesh_id not in active:
                _, _, main_vao, shadow_vao, vbo, ibo = self._mesh_cache.pop(mesh_id)
                main_vao.release()
                shadow_vao.release()
                vbo.release()
                ibo.release()
        for mesh_id in list(self._rig_cache):
            if mesh_id not in active:
                self._rig_cache.pop(mesh_id)
        active_textures = {obj.texture_path for obj in scene.objects if obj.texture_path}
        if preview_object is not None and preview_object.texture_path:
            active_textures.add(preview_object.texture_path)
        for path in list(self._texture_cache):
            if path not in active_textures:
                self._texture_cache.pop(path).release()

    def _get_texture(self, path):
        texture = self._texture_cache.get(path)
        if texture is None:
            with Image.open(path) as image:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).convert("RGBA")
                texture = self.ctx.texture(image.size, 4, image.tobytes(), alignment=1)
            texture.build_mipmaps()
            texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            self._texture_cache[path] = texture
        return texture

    def _set_shadow_map_size(self, size):
        if size not in (512, 1024, 2048, 4096):
            raise ValueError(f"Unsupported shadow map size: {size}")
        if size == self.shadow_size:
            return
        self.shadow_fbo.release()
        self.shadow_tex.release()
        self.shadow_size = size
        self.shadow_tex = self.ctx.depth_texture((size, size))
        self.shadow_fbo = self.ctx.framebuffer(depth_attachment=self.shadow_tex)

    def release(self):
        for _, _, main_vao, shadow_vao, vbo, ibo in self._mesh_cache.values():
            main_vao.release()
            shadow_vao.release()
            vbo.release()
            ibo.release()
        self._mesh_cache.clear()
        self._rig_cache.clear()
        for texture in self._texture_cache.values():
            texture.release()
        self._texture_cache.clear()
        self.shadow_fbo.release()
        self.shadow_tex.release()
        self.program.release()
        self.shadow_program.release()
        self.gizmo_program.release()
        for vao, buffer, _vertex_count in self.gizmos.values():
            vao.release()
            buffer.release()
        self.gizmos.clear()
        self.rig_vao.release()
        self.rig_buffer.release()
        if self._output_target is not None:
            framebuffer, texture, depth, _size = self._output_target
            framebuffer.release()
            texture.release()
            depth.release()
            self._output_target = None

    def render_target(self, width, height):
        size = (max(1, int(width)), max(1, int(height)))
        if self._output_target is not None and self._output_target[3] == size:
            return self._output_target[0], self._output_target[1]
        if self._output_target is not None:
            framebuffer, texture, depth, _old_size = self._output_target
            framebuffer.release()
            texture.release()
            depth.release()
        texture = self.ctx.texture(size, 4)
        depth = self.ctx.depth_renderbuffer(size)
        framebuffer = self.ctx.framebuffer(
            color_attachments=[texture], depth_attachment=depth
        )
        self._output_target = (framebuffer, texture, depth, size)
        return framebuffer, texture

    def _create_gizmo(self, name, vertices):
        buffer = self.ctx.buffer(np.asarray(vertices, dtype=np.float32).tobytes())
        vao = self.ctx.vertex_array(
            self.gizmo_program,
            [(buffer, "3f", "in_position")],
        )
        self.gizmos[name] = (vao, buffer, len(vertices))

    @staticmethod
    def _diamond_lines():
        points = np.asarray(
            ((0, 0.6, 0), (0.6, 0, 0), (0, -0.6, 0), (-0.6, 0, 0)),
            dtype=np.float32,
        )
        lines = []
        for index in range(4):
            lines.extend((points[index], points[(index + 1) % 4]))
            lines.extend((points[index], np.zeros(3, dtype=np.float32)))
        return lines

    @staticmethod
    def _arrow_lines():
        return (
            (0.0, 0.0, 0.0), (0.0, 0.0, -1.0),
            (0.0, 0.0, -1.0), (-0.16, 0.16, -0.68),
            (0.0, 0.0, -1.0), (0.16, 0.16, -0.68),
            (0.0, 0.0, -1.0), (-0.16, -0.16, -0.68),
            (0.0, 0.0, -1.0), (0.16, -0.16, -0.68),
        )

    @staticmethod
    def _spawn_lines():
        lines = []
        circle = [
            np.asarray((np.cos(angle), 0.02, np.sin(angle)), dtype=np.float32) * 0.5
            for angle in np.linspace(0, 2 * np.pi, 17)
        ]
        lines.extend(point for index in range(len(circle) - 1) for point in (circle[index], circle[index + 1]))
        lines.extend(
            (
                np.asarray((0, 0.03, 0), dtype=np.float32),
                np.asarray((0, 1, 0), dtype=np.float32),
                np.asarray((0, 1, 0), dtype=np.float32),
                np.asarray((-0.22, 0.72, 0), dtype=np.float32),
                np.asarray((0, 1, 0), dtype=np.float32),
                np.asarray((0.22, 0.72, 0), dtype=np.float32),
            )
        )
        return lines

    @staticmethod
    def _sound_lines():
        lines = [
            (-0.35, -0.22, 0), (-0.1, -0.22, 0),
            (-0.1, -0.22, 0), (0.12, -0.42, 0),
            (0.12, -0.42, 0), (0.12, 0.42, 0),
            (0.12, 0.42, 0), (-0.1, 0.22, 0),
            (-0.1, 0.22, 0), (-0.35, 0.22, 0),
            (-0.35, 0.22, 0), (-0.35, -0.22, 0),
        ]
        for radius in (0.35, 0.58, 0.8):
            points = [
                np.asarray((0.12 + radius * np.cos(angle), radius * np.sin(angle), 0), dtype=np.float32)
                for angle in np.linspace(-0.9, 0.9, 9)
            ]
            lines.extend(
                point for index in range(len(points) - 1) for point in (points[index], points[index + 1])
            )
        return lines

    @staticmethod
    def _range_lines():
        points = [
            np.asarray((np.cos(angle), 0.015, np.sin(angle)), dtype=np.float32)
            for angle in np.linspace(0, 2 * np.pi, 65)
        ]
        return [
            point for index in range(len(points) - 1)
            for point in (points[index], points[index + 1])
        ]

    @staticmethod
    def _technical_lines():
        corners = [
            np.asarray((x, y, z), dtype=np.float32)
            for x in (-0.35, 0.35)
            for y in (-0.35, 0.35)
            for z in (-0.35, 0.35)
        ]
        lines = []
        for first in range(len(corners)):
            for second in range(first + 1, len(corners)):
                if np.count_nonzero(corners[first] != corners[second]) == 1:
                    lines.extend((corners[first], corners[second]))
        lines.extend(
            (
                np.asarray((-0.55, 0, 0), dtype=np.float32),
                np.asarray((0.55, 0, 0), dtype=np.float32),
                np.asarray((0, -0.55, 0), dtype=np.float32),
                np.asarray((0, 0.55, 0), dtype=np.float32),
                np.asarray((0, 0, -0.55), dtype=np.float32),
                np.asarray((0, 0, 0.55), dtype=np.float32),
            )
        )
        return lines

    def _draw_gizmo(self, name, position, color, size, projection, view):
        vao, _buffer, vertex_count = self.gizmos[name]
        model = np.eye(4, dtype=np.float32)
        model[:3, :3] *= size
        model[:3, 3] = position
        self.gizmo_program["projection"].write(np.ascontiguousarray(projection.T).tobytes())
        self.gizmo_program["view"].write(np.ascontiguousarray(view.T).tobytes())
        self.gizmo_program["model"].write(np.ascontiguousarray(model.T).tobytes())
        self.gizmo_program["color"].value = color
        vao.render(moderngl.LINES, vertices=vertex_count)

    def _draw_editor_gizmos(self, scene, projection, view):
        self.ctx.disable(moderngl.DEPTH_TEST)
        try:
            for light in scene.lights:
                color = light.color if light.enabled else (0.4, 0.4, 0.4)
                light_position = scene.world_position(light)
                self._draw_gizmo("light", light_position, color, 0.45, projection, view)
                if light.kind == "directional":
                    direction = np.asarray(
                        scene.graph.world_direction(light, light.direction),
                        dtype=np.float32,
                    )
                    direction /= np.linalg.norm(direction) + 1e-8
                    forward = np.asarray((0.0, 0.0, -1.0), dtype=np.float32)
                    rotation = np.eye(4, dtype=np.float32)
                    cross = np.cross(forward, direction)
                    dot = float(np.dot(forward, direction))
                    if dot < -0.999:
                        rotation[:3, :3] = np.diag((1.0, -1.0, -1.0))
                    elif dot < 0.999:
                        skew = np.asarray(
                            ((0, -cross[2], cross[1]), (cross[2], 0, -cross[0]),
                             (-cross[1], cross[0], 0)),
                            dtype=np.float32,
                        )
                        rotation[:3, :3] = (
                            np.eye(3, dtype=np.float32) + skew + skew @ skew / (1.0 + dot)
                        )
                    rotation[:3, :3] *= 4.0
                    rotation[:3, 3] = light_position
                    vao, _buffer, count = self.gizmos["light_direction"]
                    self.gizmo_program["projection"].write(
                        np.ascontiguousarray(projection.T).tobytes()
                    )
                    self.gizmo_program["view"].write(np.ascontiguousarray(view.T).tobytes())
                    self.gizmo_program["model"].write(
                        np.ascontiguousarray(rotation.T).tobytes()
                    )
                    self.gizmo_program["color"].value = (0.2, 0.9, 1.0)
                    vao.render(moderngl.LINES, vertices=count)
            rig_lines = []
            for obj in scene.objects:
                if not obj.visible or not obj.rig:
                    continue
                matrix = scene.world_matrix(obj)
                bones = obj.rig["bones"]
                for bone in bones:
                    head = (matrix @ np.append(bone["head"], 1.0))[:3]
                    tail = (matrix @ np.append(self._bone_tail(bone), 1.0))[:3]
                    rig_lines.extend((head, tail))
                    parent_name = bone.get("parent")
                    if parent_name:
                        parent = next(
                            (item for item in bones if item["name"] == parent_name), None
                        )
                        if parent is not None:
                            parent_tail = (
                                matrix @ np.append(self._bone_tail(parent), 1.0)
                            )[:3]
                            rig_lines.extend((parent_tail, head))
            if rig_lines:
                line_vertices = np.asarray(rig_lines, dtype=np.float32)
                self.rig_buffer.write(line_vertices.tobytes())
                self.gizmo_program["projection"].write(
                    np.ascontiguousarray(projection.T).tobytes()
                )
                self.gizmo_program["view"].write(np.ascontiguousarray(view.T).tobytes())
                self.gizmo_program["model"].write(
                    np.ascontiguousarray(np.eye(4, dtype=np.float32).T).tobytes()
                )
                self.gizmo_program["color"].value = (1.0, 0.78, 0.25)
                self.rig_vao.render(moderngl.LINES, vertices=len(line_vertices))
            icon_types = {
                "spawn": ("spawn", (0.2, 0.95, 0.58), 0.7),
                "sound_source": ("sound", (0.35, 0.75, 1.0), 0.65),
                "logic_trigger": ("technical", (1.0, 0.68, 0.25), 0.65),
            }
            for obj in scene.objects:
                icon = icon_types.get(obj.primitive)
                if icon is not None:
                    name, color, size = icon
                    position = scene.world_position(obj)
                    self._draw_gizmo(name, position, color, size, projection, view)
                    if obj.primitive == "sound_source":
                        self._draw_gizmo(
                            "range",
                            position,
                            (0.2, 0.55, 0.9),
                            obj.sound_radius,
                            projection,
                            view,
                        )
        finally:
            self.ctx.enable(moderngl.DEPTH_TEST)

    def render_shadow_pass(self, scene, light_dir):
        light_dir = np.asarray(light_dir, dtype=np.float32)
        light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-8)

        self.shadow_fbo.use()
        self.ctx.viewport = (0, 0, self.shadow_size, self.shadow_size)
        self.ctx.clear(depth=1.0)

        light_pos = -light_dir * 12.0
        light_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        light_view = self.look_at(light_pos, light_target, up)
        light_proj = self.ortho_matrix(-12.0, 12.0, -12.0, 12.0, 0.1, 40.0)
        light_space_matrix = light_proj @ light_view

        for obj in scene.objects:
            if not obj.visible or obj.primitive in ("spawn", "sound_source", "logic_trigger"):
                continue
            vao = self.build_mesh(
                obj, program=self.shadow_program, vertices=self._deformed_vertices(obj)
            )
            model = scene.world_matrix(obj)
            self.shadow_program['light_space_matrix'].write(np.ascontiguousarray(light_space_matrix.T).tobytes())
            self.shadow_program['model'].write(np.ascontiguousarray(model.T).tobytes())
            vao.render(moderngl.TRIANGLES)

        return light_space_matrix

    @staticmethod
    def look_at(eye, center, up):
        eye = np.asarray(eye, dtype=np.float32)
        center = np.asarray(center, dtype=np.float32)
        up = np.asarray(up, dtype=np.float32)

        zaxis = eye - center
        zaxis /= np.linalg.norm(zaxis) + 1e-8
        xaxis = np.cross(up, zaxis)
        xaxis /= np.linalg.norm(xaxis) + 1e-8
        yaxis = np.cross(zaxis, xaxis)

        return np.array(
            [
                [xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, eye)],
                [yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis, eye)],
                [zaxis[0], zaxis[1], zaxis[2], -np.dot(zaxis, eye)],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def ortho_matrix(left, right, bottom, top, near, far):
        return np.array(
            [
                [2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left)],
                [0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)],
                [0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near)],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def render(
        self,
        scene,
        camera,
        width,
        height,
        show_editor_gizmos=True,
        preview_object=None,
        framebuffer=None,
    ):
        self._release_unused_meshes(scene, preview_object)
        directional = next(
            (light for light in scene.lights if light.kind == "directional" and light.enabled),
            None,
        )
        self._set_shadow_map_size(scene.shadow_map_size)
        light_dir = np.asarray(
            scene.graph.world_direction(directional, directional.direction)
            if directional else (0.0, -1.0, 0.0),
            dtype=np.float32,
        )
        if np.linalg.norm(light_dir) < 1e-8:
            raise ValueError("Scene light direction cannot be zero.")
        light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-8)
        point_lights = [
            light for light in scene.lights
            if light.kind == "point" and light.enabled
        ][:4]
        shadow_active = (
            scene.shadows_enabled
            and directional is not None
            and directional.cast_shadows
        )
        if shadow_active:
            signature = (
                self.shadow_size,
                tuple(float(value) for value in light_dir),
                tuple(
                    (
                        id(obj),
                        id(self._deformed_vertices(obj)),
                        id(obj.indices),
                        scene.world_matrix(obj).tobytes(),
                    )
                    for obj in scene.objects
                    if obj.visible
                ),
            )
            if signature != self._shadow_signature:
                self._light_space_matrix = self.render_shadow_pass(scene, light_dir)
                self._shadow_signature = signature
            light_space_matrix = self._light_space_matrix
        else:
            light_space_matrix = self._light_space_matrix
            if light_space_matrix is None:
                light_space_matrix = np.eye(4, dtype=np.float32)

        target = framebuffer if framebuffer is not None else self.ctx.screen
        target.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.clear(0.07, 0.08, 0.10, 1.0)

        projection = camera.projection_matrix(width, height)
        view = camera.view_matrix()
        self.shadow_tex.use(location=1)
        self.program["projection"].write(np.ascontiguousarray(projection.T).tobytes())
        self.program["view"].write(np.ascontiguousarray(view.T).tobytes())
        self.program["light_space_matrix"].write(
            np.ascontiguousarray(light_space_matrix.T).tobytes()
        )
        self.program["u_camera_pos"].value = tuple(camera.position)
        self.program["light_dir"].value = tuple(light_dir)
        self.program["light_color"].value = directional.color if directional else (0.0, 0.0, 0.0)
        self.program["light_intensity"].value = directional.intensity if directional else 0.0
        self.program["ambient_intensity"].value = scene.ambient_intensity
        self.program["shadows_enabled"].value = int(shadow_active)
        self.program["point_light_count"].value = len(point_lights)
        point_positions = np.zeros((4, 3), dtype=np.float32)
        point_colors = np.zeros((4, 3), dtype=np.float32)
        point_intensities = np.zeros(4, dtype=np.float32)
        for index, light in enumerate(point_lights):
            point_positions[index] = scene.world_position(light)
            point_colors[index] = light.color
            point_intensities[index] = light.intensity
        self.program["point_light_positions"].write(point_positions.tobytes())
        self.program["point_light_colors"].write(point_colors.tobytes())
        self.program["point_light_intensities"].write(point_intensities.tobytes())
        self.program["shadow_map"] = 1

        for obj in chain(
            scene.objects,
            (preview_object,) if preview_object is not None else (),
        ):
            if not obj.visible or obj.primitive in ("spawn", "sound_source", "logic_trigger"):
                continue
            vao = self.build_mesh(obj, vertices=self._deformed_vertices(obj))
            model = (
                scene.world_matrix(obj)
                if scene.graph.get_node(obj.id) is obj else obj.model_matrix()
            )
            self.program['model'].write(np.ascontiguousarray(model.T).tobytes())
            self.program['base_color'].value = obj.color
            if obj.texture_path:
                self._get_texture(obj.texture_path).use(location=2)
                self.program['surface_texture'] = 2
                self.program['has_texture'].value = 1
            else:
                self.program['has_texture'].value = 0
            self.program["texture_repeat"].value = obj.texture_repeat
            vao.render(moderngl.TRIANGLES)
        if show_editor_gizmos:
            self._draw_editor_gizmos(scene, projection, view)
