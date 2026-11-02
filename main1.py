import glfw
import moderngl
import numpy as np
import math
import time
from PIL import Image 
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

# --- МАТЕМАТИКА ---
def get_perspective_matrix(fov, aspect, near, far):
    aspect = max(aspect, 0.01) 
    f = 1.0 / math.tan(math.radians(fov) / 2.0)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ], dtype='f4')

def get_ortho_matrix(left, right, bottom, top, near, far):
    return np.array([
        [2.0 / (right - left), 0, 0, -(right + left) / (right - left)],
        [0, 2.0 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
        [0, 0, -2.0 / (far - near), -(far + near) / (far - near)],
        [0, 0, 0, 1]
    ], dtype='f4')

def look_at(eye, center, up):
    eye = np.array(eye, dtype='f4')
    center = np.array(center, dtype='f4')
    up = np.array(up, dtype='f4')
    zaxis = eye - center
    zaxis = zaxis / np.linalg.norm(zaxis)
    xaxis = np.cross(up, zaxis)
    if np.linalg.norm(xaxis) < 1e-6: # Защита от вырождения
        xaxis = np.array([1, 0, 0], dtype='f4')
    xaxis = xaxis / np.linalg.norm(xaxis)
    yaxis = np.cross(zaxis, xaxis)
    return np.array([
        [xaxis[0], yaxis[0], zaxis[0], -np.dot(xaxis, eye)],
        [xaxis[1], yaxis[1], zaxis[1], -np.dot(yaxis, eye)],
        [xaxis[2], yaxis[2], zaxis[2], -np.dot(zaxis, eye)],
        [0, 0, 0, 1]
    ], dtype='f4')

def project_to_screen(world_pos, proj, view, vp_x, vp_y, vp_w, vp_h, window_height):
    clip = proj @ view @ np.array([world_pos[0], world_pos[1], world_pos[2], 1.0])
    if clip[3] <= 0: return None
    ndc = clip[:3] / clip[3]
    
    screen_x = vp_x + (ndc[0] * 0.5 + 0.5) * vp_w
    # Точная конвертация в экранные координаты (0 - верх экрана)
    screen_y = (window_height - vp_y - vp_h) + (1.0 - (ndc[1] * 0.5 + 0.5)) * vp_h
    return np.array([screen_x, screen_y])

def point_line_distance(p, a, b):
    if np.array_equal(a, b): return np.linalg.norm(p - a)
    # 3D-векторы для np.cross (фикс NumPy 2.0)
    p3 = np.array([p[0], p[1], 0.0]); a3 = np.array([a[0], a[1], 0.0]); b3 = np.array([b[0], b[1], 0.0])
    return np.linalg.norm(np.cross(b3 - a3, a3 - p3)) / np.linalg.norm(b3 - a3)

def load_texture(ctx, filepath):
    img = Image.open(filepath).transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA")
    texture = ctx.texture(img.size, 4, img.tobytes())
    texture.build_mipmaps()
    texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    return texture

def create_floor_texture(ctx, size=512, squares=16):
    img = np.zeros((size, size, 3), dtype='u1')
    step = size // squares
    for y in range(size):
        for x in range(size):
            img[y, x] = [80, 80, 80] if (x // step + y // step) % 2 == 0 else [120, 120, 120]
    texture = ctx.texture((size, size), 3, img.tobytes())
    texture.build_mipmaps()
    texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    texture.repeat_x = True; texture.repeat_y = True
    return texture

def main():
    if not glfw.init(): return
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

    window = glfw.create_window(1280, 720, "DEc | Рендерер прототипа", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(0) 

    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.enable(moderngl.CULL_FACE)
    imgui.create_context()
    impl = GlfwRenderer(window)

    # --- ШЕЙДЕРЫ ---
    depth_prog = ctx.program(vertex_shader="""#version 330\nin vec3 in_position; uniform mat4 light_space_matrix; uniform mat4 model;\nvoid main() { gl_Position = light_space_matrix * model * vec4(in_position, 1.0); }""", fragment_shader="""#version 330\nvoid main() { }""")
    prog = ctx.program(
        vertex_shader="""#version 330\nin vec3 in_position; in vec2 in_uv; in vec3 in_normal; uniform mat4 mvp; uniform mat4 model; uniform mat4 light_space_matrix; out vec2 v_uv; out vec3 v_normal; out vec4 v_frag_pos_light_space; out vec3 v_frag_pos_world;\nvoid main() { gl_Position = mvp * vec4(in_position, 1.0); v_uv = in_uv; v_normal = mat3(model) * in_normal; v_frag_pos_world = vec3(model * vec4(in_position, 1.0)); v_frag_pos_light_space = light_space_matrix * vec4(v_frag_pos_world, 1.0); }""",
        fragment_shader="""#version 330\nin vec2 v_uv; in vec3 v_normal; in vec4 v_frag_pos_light_space; uniform sampler2D tex; uniform sampler2D shadow_map; uniform vec3 light_dir; out vec4 f_color;\nfloat calculate_shadow(vec4 frag_pos_light_space, vec3 normal, vec3 light_dir) { vec3 proj_coords = frag_pos_light_space.xyz / frag_pos_light_space.w; proj_coords = proj_coords * 0.5 + 0.5; if(proj_coords.z > 1.0) return 0.0; float current_depth = proj_coords.z; float bias = max(0.05 * (1.0 - dot(normal, light_dir)), 0.005); float shadow = 0.0; vec2 texel_size = 1.0 / textureSize(shadow_map, 0); for(int x = -1; x <= 1; ++x) for(int y = -1; y <= 1; ++y) { float pcf_depth = texture(shadow_map, proj_coords.xy + vec2(x, y) * texel_size).r; shadow += current_depth - bias > pcf_depth ? 0.4 : 1.0; } return shadow / 9.0; }\nvoid main() { vec4 tex_color = texture(tex, v_uv); float diff = max(dot(normalize(v_normal), normalize(light_dir)), 0.0); float shadow = calculate_shadow(v_frag_pos_light_space, normalize(v_normal), normalize(light_dir)); f_color = vec4(tex_color.rgb * (0.3 + (diff * 0.7 * shadow)), tex_color.a); }"""
    )
    grid_prog = ctx.program(vertex_shader="""#version 330\nin vec3 in_position; uniform mat4 mvp;\nvoid main() { gl_Position = mvp * vec4(in_position, 1.0); }""", fragment_shader="""#version 330\nout vec4 f_color;\nvoid main() { f_color = vec4(0.5, 0.5, 0.5, 1.0); }""")
    color_prog = ctx.program(vertex_shader="""#version 330\nin vec3 in_position; uniform mat4 mvp;\nvoid main() { gl_Position = mvp * vec4(in_position, 1.0); }""", fragment_shader="""#version 330\nuniform vec3 color; out vec4 f_color;\nvoid main() { f_color = vec4(color, 1.0); }""")

    # --- ГЕОМЕТРИЯ ---
    v_cube = [-0.5,-0.5,0.5,0,0,0,0,1, 0.5,-0.5,0.5,1,0,0,0,1, 0.5,0.5,0.5,1,1,0,0,1, -0.5,-0.5,0.5,0,0,0,0,1, 0.5,0.5,0.5,1,1,0,0,1, -0.5,0.5,0.5,0,1,0,0,1,
                 0.5,-0.5,-0.5,0,0,0,0,-1, -0.5,-0.5,-0.5,1,0,0,0,-1, -0.5,0.5,-0.5,1,1,0,0,-1, 0.5,-0.5,-0.5,0,0,0,0,-1, -0.5,0.5,-0.5,1,1,0,0,-1, 0.5,0.5,-0.5,0,1,0,0,-1,
                 -0.5,-0.5,-0.5,0,0,-1,0,0, -0.5,-0.5,0.5,1,0,-1,0,0, -0.5,0.5,0.5,1,1,-1,0,0, -0.5,-0.5,-0.5,0,0,-1,0,0, -0.5,0.5,0.5,1,1,-1,0,0, -0.5,0.5,-0.5,0,1,-1,0,0,
                 0.5,-0.5,0.5,0,0,1,0,0, 0.5,-0.5,-0.5,1,0,1,0,0, 0.5,0.5,-0.5,1,1,1,0,0, 0.5,-0.5,0.5,0,0,1,0,0, 0.5,0.5,-0.5,1,1,1,0,0, 0.5,0.5,0.5,0,1,1,0,0,
                 -0.5,0.5,0.5,0,0,0,1,0, 0.5,0.5,0.5,1,0,0,1,0, 0.5,0.5,-0.5,1,1,0,1,0, -0.5,0.5,0.5,0,0,0,1,0, 0.5,0.5,-0.5,1,1,0,1,0, -0.5,0.5,-0.5,0,1,0,1,0,
                 0.5,-0.5,0.5,0,0,0,-1,0, -0.5,-0.5,0.5,1,0,0,-1,0, -0.5,-0.5,-0.5,1,1,0,-1,0, 0.5,-0.5,0.5,0,0,0,-1,0, -0.5,-0.5,-0.5,1,1,0,-1,0, 0.5,-0.5,-0.5,0,1,0,-1,0]
    
    vbo = ctx.buffer(np.array(v_cube, dtype='f4').tobytes())
    vao = ctx.vertex_array(prog, [(vbo, '3f 2f 3f', 'in_position', 'in_uv', 'in_normal')])
    depth_vao_cube = ctx.vertex_array(depth_prog, [(vbo, '3f 20x', 'in_position')])
    line_vbo = ctx.buffer(reserve=1024 * 3 * 4) 
    line_vao = ctx.vertex_array(color_prog, [(line_vbo, '3f', 'in_position')])

    v_floor = [-5.0, -0.5, -5.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, -0.5, -5.0, 5.0, 0.0, 0.0, 1.0, 0.0, 5.0, -0.5, 5.0, 5.0, 5.0, 0.0, 1.0, 0.0, -5.0, -0.5, -5.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, -0.5, 5.0, 5.0, 5.0, 0.0, 1.0, 0.0, -5.0, -0.5, 5.0, 0.0, 5.0, 0.0, 1.0, 0.0]
    floor_vbo = ctx.buffer(np.array(v_floor, dtype='f4').tobytes())
    floor_vao = ctx.vertex_array(prog, [(floor_vbo, '3f 2f 3f', 'in_position', 'in_uv', 'in_normal')])

    grid_lines = []
    for i in range(-10, 11):
        grid_lines.extend([-10.0, -0.49, float(i), 10.0, -0.49, float(i)])
        grid_lines.extend([float(i), -0.49, -10.0, float(i), -0.49, 10.0])
    grid_vbo = ctx.buffer(np.array(grid_lines, dtype='f4').tobytes())
    grid_vao = ctx.vertex_array(grid_prog, [(grid_vbo, '3f', 'in_position')])

    try: tex = load_texture(ctx, "e621.png") 
    except FileNotFoundError: tex = create_floor_texture(ctx)
    floor_tex = create_floor_texture(ctx)

    shadow_size = 2048
    shadow_tex = ctx.depth_texture((shadow_size, shadow_size))
    shadow_fbo = ctx.framebuffer(depth_attachment=shadow_tex)
    prog['shadow_map'] = 2 

    # --- СОСТОЯНИЕ ---
    light_dir = [0.5, 1.0, 0.5]
    wireframe_mode = False
    fixed_ui = True 
    cube_pos = np.array([0.0, 0.0, 0.0], dtype='f4')
    camera_obj_pos = np.array([5.0, 5.0, 5.0], dtype='f4') 
    selected_obj = 'cube' 
    
    # --- ИДЕАЛЬНЫЙ СПАВН 3D КАМЕРЫ ---
    # Позиция [10, 10, 10], смотрит ровно в центр [0,0,0]
    cam_pos = np.array([10.0, 10.0, 10.0], dtype='f4')
    cam_yaw = math.radians(-45.0)
    cam_pitch = math.radians(-35.264) # Точный угол, чтобы смотреть в [0,0,0]
    cam_speed = 10.0
    last_mouse_x, last_mouse_y = 0.0, 0.0

    active_axis, is_dragging = None, False
    drag_start_mouse = np.array([0.0, 0.0])
    drag_start_pos = np.array([0.0, 0.0, 0.0])
    ortho_zoom = 6.0 

    target_fps = 200.0
    target_frame_time = 1.0 / target_fps
    fps_timer, frames_rendered, real_fps = time.perf_counter(), 0, 0.0
    last_frame_time = time.perf_counter()

    while not glfw.window_should_close(window):
        frame_start = time.perf_counter()
        dt = frame_start - last_frame_time
        last_frame_time = frame_start

        glfw.poll_events()
        impl.process_inputs()

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        width, height = glfw.get_framebuffer_size(window)
        if width == 0 or height == 0: continue 
        w_half, h_half = width // 2, height // 2
        mouse_x, mouse_y = glfw.get_cursor_pos(window)
        io = imgui.get_io()
        scroll = io.mouse_wheel
        io.mouse_wheel = 0.0

        # --- УПРАВЛЕНИЕ 3D КАМЕРОЙ (WASD + MOUSE) ---
        in_3d_view = (mouse_x < w_half and mouse_y < h_half)
        
        if not io.want_capture_mouse:
            # Вращение ПКМ (Теперь стандартное, без инверсии)
            if in_3d_view and glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
                cam_yaw += (mouse_x - last_mouse_x) * 0.005
                cam_pitch -= (mouse_y - last_mouse_y) * 0.005
                cam_pitch = max(math.radians(-89), min(math.radians(89), cam_pitch))
            
            # Вектора движения
            forward = np.array([math.cos(cam_pitch)*math.sin(cam_yaw), math.sin(cam_pitch), -math.cos(cam_pitch)*math.cos(cam_yaw)], dtype='f4')
            right = np.array([math.cos(cam_yaw), 0, math.sin(cam_yaw)], dtype='f4')
            up = np.cross(right, forward)

            move_vec = np.zeros(3, dtype='f4')
            if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS: move_vec += forward
            if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS: move_vec -= forward
            if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS: move_vec += right
            if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS: move_vec -= right
            if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS: move_vec += up
            if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS: move_vec -= up
            
            if np.linalg.norm(move_vec) > 0:
                cam_pos += move_vec * cam_speed * dt

            # Зум / Скорость
            if scroll != 0:
                if in_3d_view: cam_speed = max(1.0, cam_speed + scroll * 2.0)
                else: ortho_zoom = max(1.0, ortho_zoom - scroll * 0.5)

        last_mouse_x, last_mouse_y = mouse_x, mouse_y

        # --- ИНТЕРФЕЙС ---
        imgui.new_frame()
        if fixed_ui: imgui.set_next_window_pos((width - 10, 10), imgui.Cond_.always, (1.0, 0.0))
        imgui.begin("Dev Tools", None, imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.always_auto_resize)
        imgui.text(f"FPS: {real_fps:.1f}")
        imgui.separator()
        imgui.text("Select Object:")
        if imgui.radio_button("Cube", selected_obj == 'cube'): selected_obj = 'cube'
        imgui.same_line()
        if imgui.radio_button("Camera Obj", selected_obj == 'camera'): selected_obj = 'camera'
        imgui.separator()
        imgui.text("Light Direction:")
        imgui.push_item_width(150)
        _, light_dir[0] = imgui.slider_float("X", light_dir[0], -2.0, 2.0)
        _, light_dir[1] = imgui.slider_float("Y", light_dir[1], -2.0, 2.0)
        _, light_dir[2] = imgui.slider_float("Z", light_dir[2], -2.0, 2.0)
        imgui.pop_item_width()
        imgui.separator()
        _, wireframe_mode = imgui.checkbox("Wireframe Mode (3D)", wireframe_mode)
        _, fixed_ui = imgui.checkbox("Fix Position", fixed_ui)
        imgui.end()

        # --- МАТРИЦЫ КАМЕР ---
        view_persp = look_at(cam_pos, cam_pos + forward, up)
        view_top   = look_at([0, 0, 10], [0, 0, 0], [0, 1, 0])
        view_front = look_at([10, 0, 0], [0, 0, 0], [0, 1, 0])
        view_side  = look_at([0, 10, 0], [0, 0, 0], [0, 0, -1])

        proj_persp = get_perspective_matrix(45.0, w_half/max(1,h_half), 0.1, 100.0)
        proj_ortho = get_ortho_matrix(-ortho_zoom*(w_half/max(1,h_half)), ortho_zoom*(w_half/max(1,h_half)), -ortho_zoom, ortho_zoom, 0.1, 100.0)

        # --- ЛОГИКА ВЗАИМОДЕЙСТВИЯ С ГИЗМО ---
        if not io.want_capture_mouse:
            target_obj_pos = cube_pos if selected_obj == 'cube' else camera_obj_pos
            
            if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS and not is_dragging:
                is_dragging = True
                drag_start_mouse = np.array([mouse_x, mouse_y])
                drag_start_pos = np.copy(target_obj_pos)
                active_axis = None
                
                view_id = None
                vp = None
                if mouse_x < w_half and mouse_y < h_half: view_id, vp = "3d", (0, h_half, w_half, h_half)
                elif mouse_x >= w_half and mouse_y < h_half: view_id, vp = "top", (w_half, h_half, w_half, h_half)
                elif mouse_x < w_half and mouse_y >= h_half: view_id, vp = "front", (0, 0, w_half, h_half)
                elif mouse_x >= w_half and mouse_y >= h_half: view_id, vp = "side", (w_half, 0, w_half, h_half)

                if view_id:
                    proj, view = (proj_persp, view_persp) if view_id == "3d" else (proj_ortho, {'top':view_top, 'front':view_front, 'side':view_side}[view_id])
                    
                    axes = [('X', [1,0,0]), ('Y', [0,1,0]), ('Z', [0,0,1])]
                    if view_id == "top": axes = [('X', [1,0,0]), ('Y', [0,1,0])]
                    elif view_id == "front": axes = [('Y', [0,1,0]), ('Z', [0,0,1])]
                    elif view_id == "side": axes = [('X', [1,0,0]), ('Z', [0,0,1])]

                    center_screen = project_to_screen(target_obj_pos, proj, view, *vp, height)
                    if center_screen is not None:
                        for axis, offset in axes:
                            arrow_len = 2.0 if view_id == "3d" else 1.0
                            end_screen = project_to_screen(target_obj_pos + np.array(offset)*arrow_len, proj, view, *vp, height)
                            if end_screen is not None and point_line_distance(np.array([mouse_x, mouse_y]), center_screen, end_screen) < 15:
                                active_axis = axis
                                break
                if active_axis is None: is_dragging = False
            
            elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.RELEASE:
                is_dragging, active_axis = False, None

            if is_dragging and active_axis is not None:
                dx, dy = mouse_x - drag_start_mouse[0], mouse_y - drag_start_mouse[1]
                axis_dir = {'X': np.array([1,0,0]), 'Y': np.array([0,1,0]), 'Z': np.array([0,0,1])}[active_axis]
                
                # Вычисляем экранный вектор направления стрелки
                p0 = project_to_screen(target_obj_pos, proj, view, *vp, height)
                p1 = project_to_screen(target_obj_pos + axis_dir, proj, view, *vp, height)
                
                if p0 is not None and p1 is not None:
                    screen_dir = p1 - p0
                    if np.linalg.norm(screen_dir) > 1e-6:
                        screen_dir /= np.linalg.norm(screen_dir)
                        # Проекция движения мыши на экранный вектор стрелки (без инверсий)
                        move_amount = np.dot([dx, dy], screen_dir)
                        
                        if view_id == "3d":
                            move_amount *= 0.05 * np.linalg.norm(cam_pos - target_obj_pos)
                        else:
                            move_amount *= (2.0 * ortho_zoom) / h_half
                            
                        target_obj_pos[:] = drag_start_pos + axis_dir * move_amount

        # --- 1 ПРОХОД: ТЕНИ ---
        light_space_matrix = get_ortho_matrix(-5, 5, -5, 5, 0.1, 30.0) @ look_at(np.array(light_dir)*10.0, [0, 0, 0], [0, 1, 0])
        shadow_fbo.use()
        ctx.viewport = (0, 0, shadow_size, shadow_size)
        ctx.clear()
        t = frame_start * 0.5 
        cos_y, sin_y = math.cos(t), math.sin(t)
        cube_model = np.array([[cos_y,0,sin_y,cube_pos[0]], [0,1,0,cube_pos[1]+0.5], [-sin_y,0,cos_y,cube_pos[2]], [0,0,0,1]], dtype='f4')

        depth_prog['light_space_matrix'].write(light_space_matrix.T.tobytes())
        depth_prog['model'].write(cube_model.T.tobytes())
        ctx.wireframe = False
        depth_vao_cube.render(moderngl.TRIANGLES)

        # --- 2 ПРОХОД: ОКНА ---
        ctx.screen.use()
        ctx.clear(0.1, 0.1, 0.12, 1.0) 
        prog['light_dir'].value = tuple(light_dir)
        prog['light_space_matrix'].write(light_space_matrix.T.tobytes())
        shadow_tex.use(2)

        def draw_viewport(vp_x, vp_y, vp_w, vp_h, view_mat, proj_mat, is_ortho=False, view_id="3d"):
            ctx.scissor = (vp_x, vp_y, vp_w, vp_h)
            ctx.viewport = (vp_x, vp_y, vp_w, vp_h)
            ctx.clear(0.15, 0.15, 0.15, 1.0) if is_ortho else ctx.clear(0.2, 0.2, 0.25, 1.0)
            
            if not is_ortho:
                ctx.disable(moderngl.CULL_FACE)
                prog['mvp'].write((proj_mat @ view_mat).T.tobytes())
                prog['model'].write(np.eye(4, dtype='f4').T.tobytes())
                floor_tex.use(0); prog['tex'] = 0
                floor_vao.render(moderngl.TRIANGLES)

            ctx.disable(moderngl.CULL_FACE)
            grid_prog['mvp'].write((proj_mat @ view_mat).T.tobytes())
            grid_vao.render(moderngl.LINES)

            ctx.enable(moderngl.CULL_FACE)
            prog['mvp'].write((proj_mat @ view_mat @ cube_model).T.tobytes())
            prog['model'].write(cube_model.T.tobytes())
            tex.use(0); prog['tex'] = 0
            ctx.wireframe = True if is_ortho else wireframe_mode
            vao.render(moderngl.TRIANGLES)
            ctx.wireframe = False

            ctx.disable(moderngl.CULL_FACE)
            color_prog['mvp'].write((proj_mat @ view_mat).T.tobytes())
            
            # Отрисовка визуализации камеры (объекта)
            cam_lines = [camera_obj_pos[0],camera_obj_pos[1],camera_obj_pos[2], camera_obj_pos[0]+0.5,camera_obj_pos[1]+0.5,camera_obj_pos[2]+0.5, camera_obj_pos[0],camera_obj_pos[1],camera_obj_pos[2], camera_obj_pos[0]-0.5,camera_obj_pos[1]+0.5,camera_obj_pos[2]+0.5, camera_obj_pos[0],camera_obj_pos[1],camera_obj_pos[2], camera_obj_pos[0]+0.5,camera_obj_pos[1]-0.5,camera_obj_pos[2]+0.5, camera_obj_pos[0],camera_obj_pos[1],camera_obj_pos[2], camera_obj_pos[0]-0.5,camera_obj_pos[1]-0.5,camera_obj_pos[2]+0.5]
            line_vbo.write(np.array(cam_lines, dtype='f4').tobytes())
            color_prog['color'].value = (1.0, 1.0, 0.0) 
            line_vao.render(moderngl.LINES, vertices=8)

            # Отрисовка реальной позиции камеры (чтобы видеть откуда смотрим)
            if view_id == "3d":
                color_prog['color'].value = (0.0, 1.0, 1.0) # Голубой
                real_cam_lines = [cam_pos[0],cam_pos[1],cam_pos[2], cam_pos[0]+forward[0]*1.0,cam_pos[1]+forward[1]*1.0,cam_pos[2]+forward[2]*1.0]
                line_vbo.write(np.array(real_cam_lines, dtype='f4').tobytes())
                line_vao.render(moderngl.LINES, vertices=2)

            # Гизмо
            target_p = cube_pos if selected_obj == 'cube' else camera_obj_pos
            if view_id == "3d": gizmos = [('X', (1,0,0), [2.0,0,0]), ('Y', (0,1,0), [0,2.0,0]), ('Z', (0,0,1), [0,0,2.0])]
            elif view_id == "top": gizmos = [('X', (1,0,0), [1.0,0,0]), ('Y', (0,1,0), [0,1.0,0])]
            elif view_id == "front": gizmos = [('Y', (0,1,0), [0,1.0,0]), ('Z', (0,0,1), [0,0,1.0])]
            elif view_id == "side": gizmos = [('X', (1,0,0), [1.0,0,0]), ('Z', (0,0,1), [0,0,1.0])]

            for axis, color, offset in gizmos:
                c = (1, 1, 0) if active_axis == axis else color
                color_prog['color'].value = c
                g_lines = [target_p[0], target_p[1], target_p[2], target_p[0]+offset[0], target_p[1]+offset[1], target_p[2]+offset[2]]
                line_vbo.write(np.array(g_lines, dtype='f4').tobytes())
                line_vao.render(moderngl.LINES, vertices=2)

        draw_viewport(0, h_half, w_half, h_half, view_persp, proj_persp, is_ortho=False, view_id="3d") 
        draw_viewport(w_half, h_half, w_half, h_half, view_top, proj_ortho, is_ortho=True, view_id="top")    
        draw_viewport(0, 0, w_half, h_half, view_front, proj_ortho, is_ortho=True, view_id="front")            
        draw_viewport(w_half, 0, w_half, h_half, view_side, proj_ortho, is_ortho=True, view_id="side")        

        ctx.scissor = None
        ctx.viewport = (0, 0, width, height)

        draw_list = imgui.get_background_draw_list()
        draw_list.add_line((w_half, 0), (w_half, height), 0xFF555555, 2.0)
        draw_list.add_line((0, h_half), (width, h_half), 0xFF555555, 2.0)
        draw_list.add_text((10, 10), 0xFFFFFFFF, "camera (3D) [WASD + RMB]")
        draw_list.add_text((w_half + 10, 10), 0xFFFFFFFF, "top (x/y)")
        draw_list.add_text((10, h_half + 10), 0xFFFFFFFF, "front (y/z)")
        draw_list.add_text((w_half + 10, h_half + 10), 0xFFFFFFFF, "side (x/z)")

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

        elapsed = time.perf_counter() - frame_start
        if target_frame_time - elapsed > 0: time.sleep(target_frame_time - elapsed)
        
        frames_rendered += 1
        if time.perf_counter() - fps_timer >= 1.0:
            real_fps = frames_rendered / (time.perf_counter() - fps_timer)
            frames_rendered, fps_timer = 0, time.perf_counter()

    impl.shutdown()
    glfw.terminate()

if __name__ == "__main__":
    main()