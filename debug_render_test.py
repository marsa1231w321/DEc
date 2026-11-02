import time

import glfw
import moderngl
import numpy as np

if not glfw.init():
    raise SystemExit('glfw init failed')

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
window = glfw.create_window(640, 480, 'debug', None, None)
if not window:
    glfw.terminate()
    raise SystemExit('window failed')

glfw.make_context_current(window)
ctx = moderngl.create_context()
prog = ctx.program(
    vertex_shader='''#version 330
        in vec2 in_position;
        void main() { gl_Position = vec4(in_position, 0.0, 1.0); }
    ''',
    fragment_shader='''#version 330
        out vec4 color;
        void main() { color = vec4(1.0, 0.0, 0.0, 1.0); }
    '''
)
tri = np.array([0.0, 0.5, -0.5, -0.5, 0.5, -0.5], dtype='f4')
vbo = ctx.buffer(tri.tobytes())
vao = ctx.vertex_array(prog, [(vbo, '2f', 'in_position')])
ctx.viewport = (0, 0, 640, 480)
ctx.clear(0.1, 0.1, 0.1, 1.0)
vao.render()
glfw.swap_buffers(window)
print('triangle rendered once')
time.sleep(2)
glfw.destroy_window(window)
glfw.terminate()
