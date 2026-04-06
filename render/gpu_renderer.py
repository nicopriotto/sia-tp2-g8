import logging
import os

import numpy as np

from render.renderer import Renderer

logger = logging.getLogger(__name__)


def _ctx_kwargs() -> dict:
    """Retorna kwargs para create_standalone_context segun el entorno."""
    backend = os.environ.get("MODERNGL_BACKEND")
    if backend:
        return {"backend": backend}
    # Autodetectar: si no hay display (servidor/Colab), intentar EGL
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return {"backend": "egl"}
    return {}


def gpu_available() -> bool:
    """Retorna True si ModernGL esta instalado y puede crear un contexto offscreen."""
    try:
        import moderngl
        ctx = moderngl.create_standalone_context(**_ctx_kwargs())
        ctx.release()
        return True
    except Exception:
        return False


class GPURenderer(Renderer):
    """
    Renderer que usa ModernGL para renderizar triangulos en la GPU
    y calcular fitness directamente en shaders.
    """

    def __init__(self, width: int, height: int, target_image: np.ndarray):
        import moderngl

        self.width = width
        self.height = height
        self.ctx = moderngl.create_standalone_context(**_ctx_kwargs())

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (
            moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA,
        )

        self._load_shaders()

        # Framebuffer offscreen para renderizado de triangulos
        self.render_texture = self.ctx.texture((width, height), 4, dtype='f4')
        self.render_fbo = self.ctx.framebuffer(color_attachments=[self.render_texture])

        # Imagen objetivo en VRAM (una sola vez)
        # Las texturas OpenGL tienen Y=0 abajo; como tambien invertimos Y
        # en el vertex shader, ambas imagenes quedan consistentes.
        # Se carga flipud para coincidir con el layout del framebuffer.
        target_flipped = np.flipud(target_image).copy()
        self.target_texture = self.ctx.texture(
            (width, height), 4,
            data=target_flipped.astype(np.float32).tobytes(),
            dtype='f4',
        )

        self._init_fitness_pipeline()

        logger.info("GPURenderer inicializado: %dx%d, backend ModernGL", width, height)

    def _load_shaders(self):
        shader_dir = os.path.join(os.path.dirname(__file__), "shaders")

        def _read(filename):
            with open(os.path.join(shader_dir, filename), "r") as f:
                return f.read()

        vert_src = _read("triangle.vert")
        frag_src = _read("triangle.frag")
        self.triangle_program = self.ctx.program(
            vertex_shader=vert_src,
            fragment_shader=frag_src,
        )

        fullscreen_vert = """
        #version 330 core
        in vec2 in_position;
        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
        }
        """

        self.fitness_mse_program = self.ctx.program(
            vertex_shader=fullscreen_vert,
            fragment_shader=_read("fitness_mse.frag"),
        )

        self.fitness_mae_program = self.ctx.program(
            vertex_shader=fullscreen_vert,
            fragment_shader=_read("fitness_mae.frag"),
        )

        self.reduce_program = self.ctx.program(
            vertex_shader=fullscreen_vert,
            fragment_shader=_read("reduce.frag"),
        )

        quad_vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype='f4')
        self.quad_vbo = self.ctx.buffer(quad_vertices)

    def _init_fitness_pipeline(self):
        """Crea las texturas y framebuffers para la cadena de reduccion."""
        w, h = self.width, self.height
        self.reduction_levels = []

        while w > 1 or h > 1:
            w = max(1, w // 2)
            h = max(1, h // 2)
            tex = self.ctx.texture((w, h), 4, dtype='f4')
            fbo = self.ctx.framebuffer(color_attachments=[tex])
            self.reduction_levels.append((tex, fbo, w, h))

        # Textura de error (mismo tamano que la imagen)
        self.error_texture = self.ctx.texture(
            (self.width, self.height), 4, dtype='f4',
        )
        self.error_fbo = self.ctx.framebuffer(color_attachments=[self.error_texture])

    def render(self, genes: np.ndarray, width: int = None, height: int = None, gene_type: str = "triangle") -> np.ndarray:
        """
        Renderiza un array de genes en la GPU (solo triangulos).

        Retorna imagen como numpy array float32 (H, W, 4) en [0, 1].
        """
        w = width or self.width
        h = height or self.height

        self.render_fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(1.0, 1.0, 1.0, 1.0)

        if len(genes) > 0:
            # Filtrar activos: columna 10 > 0.5
            active_mask = genes[:, 10] > 0.5
            active = genes[active_mask]

            if len(active) > 0:
                n = active.shape[0]
                # Vectorized packing: cada triangulo = 3 vertices x 6 floats (x, y, r, g, b, a)
                vertex_data = np.empty((n, 3, 6), dtype='f4')
                # Coordenadas de vertices
                for v, (xi, yi) in enumerate([(0, 1), (2, 3), (4, 5)]):
                    vertex_data[:, v, 0] = active[:, xi]
                    vertex_data[:, v, 1] = active[:, yi]
                # Color normalizado (compartido entre los 3 vertices)
                for v in range(3):
                    vertex_data[:, v, 2] = active[:, 6] / 255.0  # r
                    vertex_data[:, v, 3] = active[:, 7] / 255.0  # g
                    vertex_data[:, v, 4] = active[:, 8] / 255.0  # b
                    vertex_data[:, v, 5] = active[:, 9]           # a

                vbo = self.ctx.buffer(vertex_data.reshape(-1))
                vao = self.ctx.vertex_array(
                    self.triangle_program,
                    [(vbo, '2f 4f', 'in_position', 'in_color')],
                )
                vao.render(mode=self.ctx.TRIANGLES)
                vbo.release()
                vao.release()

        raw = self.render_fbo.read(components=4, dtype='f4')
        image = np.frombuffer(raw, dtype=np.float32).reshape((h, w, 4))
        image = np.flipud(image)
        return image.copy()

    def compute_fitness(self, genes: np.ndarray, fitness_type: str = "mse", gene_type: str = "triangle") -> float:
        """
        Renderiza los triangulos y calcula el fitness enteramente en GPU.

        Returns:
            Fitness en rango (0, 1].
        """
        # Paso 0: Renderizar triangulos
        self.render(genes, gene_type=gene_type)

        # Paso 1: Calcular error por pixel
        self.error_fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)

        program = self.fitness_mse_program if fitness_type == "mse" else self.fitness_mae_program

        self.render_texture.use(location=0)
        self.target_texture.use(location=1)
        program['u_generated'].value = 0
        program['u_target'].value = 1

        vao = self.ctx.vertex_array(
            program,
            [(self.quad_vbo, '2f', 'in_position')],
        )
        vao.render(mode=self.ctx.TRIANGLE_STRIP)
        vao.release()

        # Paso 2: Reduccion iterativa
        prev_texture = self.error_texture
        for tex, fbo, w, h in self.reduction_levels:
            fbo.use()
            self.ctx.viewport = (0, 0, w, h)
            self.ctx.clear(0.0, 0.0, 0.0, 0.0)

            prev_texture.use(location=0)
            self.reduce_program['u_input'].value = 0

            prev_w = prev_texture.size[0]
            prev_h = prev_texture.size[1]
            self.reduce_program['u_texel_size'].value = (1.0 / prev_w, 1.0 / prev_h)

            reduce_vao = self.ctx.vertex_array(
                self.reduce_program,
                [(self.quad_vbo, '2f', 'in_position')],
            )
            reduce_vao.render(mode=self.ctx.TRIANGLE_STRIP)
            reduce_vao.release()

            prev_texture = tex

        # Restaurar viewport
        self.ctx.viewport = (0, 0, self.width, self.height)

        # Paso 3: Leer el pixel 1x1 final
        _, last_fbo, _, _ = self.reduction_levels[-1]
        raw = last_fbo.read(components=4, dtype='f4')
        error_rgba = np.frombuffer(raw, dtype=np.float32)

        mean_error = float(np.mean(error_rgba[:3]))
        fitness = 1.0 / (1.0 + mean_error)
        return fitness

    @staticmethod
    def detect_device(preference: str = "auto") -> str:
        """Detecta el dispositivo GPU disponible."""
        try:
            import moderngl
            ctx = moderngl.create_standalone_context(**_ctx_kwargs())
            info = ctx.info
            vendor = info.get("GL_VENDOR", "").lower()
            renderer_name = info.get("GL_RENDERER", "").lower()
            ctx.release()

            is_dedicated = any(
                kw in vendor or kw in renderer_name
                for kw in ["nvidia", "amd", "radeon", "geforce", "rtx", "gtx"]
            )
            is_integrated = any(
                kw in vendor or kw in renderer_name
                for kw in ["intel", "mesa", "llvmpipe"]
            )

            device_type = "dedicated" if is_dedicated else "integrated" if is_integrated else "unknown"

            if preference == "dedicated" and not is_dedicated:
                logger.warning("GPU dedicada solicitada pero no encontrada. Disponible: %s", renderer_name)

            return f"{device_type}: {renderer_name}"

        except Exception as e:
            logger.warning("No se pudo detectar GPU: %s", e)
            return "none"

    def release(self):
        """Libera todos los recursos de GPU."""
        self.render_texture.release()
        self.render_fbo.release()
        self.target_texture.release()
        self.error_texture.release()
        self.error_fbo.release()
        for tex, fbo, _, _ in self.reduction_levels:
            tex.release()
            fbo.release()
        self.quad_vbo.release()
        self.triangle_program.release()
        self.fitness_mse_program.release()
        self.fitness_mae_program.release()
        self.reduce_program.release()
        self.ctx.release()
        logger.info("GPURenderer: recursos liberados")

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
