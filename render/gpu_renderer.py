import logging
import os
import platform
from contextlib import contextmanager

import numpy as np

from render.renderer import Renderer

logger = logging.getLogger(__name__)

DEDICATED_OFFLOAD_ENV = {
    "__NV_PRIME_RENDER_OFFLOAD": "1",
    "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
}

DEDICATED_MARKERS = (
    "nvidia",
    "geforce",
    "rtx",
    "gtx",
    "quadro",
    "tesla",
    "radeon rx",
    "firepro",
    "radeon pro",
)

INTEGRATED_MARKERS = (
    "intel",
    "iris",
    "uhd",
    "renoir",
    "cezanne",
    "rembrandt",
    "phoenix",
    "raven",
    "radeon graphics",
    "apu",
)

SOFTWARE_MARKERS = (
    "llvmpipe",
    "softpipe",
    "software rasterizer",
)


@contextmanager
def _temporary_env(overrides: dict[str, str | None]):
    """Aplica overrides de entorno solo durante la creacion del contexto."""
    previous: dict[str, str] = {}
    missing: set[str] = set()

    for key, value in overrides.items():
        if key in os.environ:
            previous[key] = os.environ[key]
        else:
            missing.add(key)

        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    try:
        yield
    finally:
        for key, value in previous.items():
            os.environ[key] = value
        for key in missing:
            os.environ.pop(key, None)


def _context_env_overrides(preference: str) -> dict[str, str | None]:
    if platform.system().lower() != "linux":
        return {}

    if preference == "dedicated":
        return DEDICATED_OFFLOAD_ENV.copy()

    if preference == "integrated":
        return {key: None for key in DEDICATED_OFFLOAD_ENV}

    return {}


def _classify_device_info(info: dict) -> tuple[str, str]:
    vendor = str(info.get("GL_VENDOR", "") or "")
    renderer_name = str(info.get("GL_RENDERER", "") or "") or vendor or "unknown"
    combined = f"{vendor} {renderer_name}".lower()

    if any(marker in combined for marker in SOFTWARE_MARKERS):
        return "unknown", renderer_name

    if any(marker in combined for marker in INTEGRATED_MARKERS):
        return "integrated", renderer_name

    if any(marker in combined for marker in DEDICATED_MARKERS):
        return "dedicated", renderer_name

    return "unknown", renderer_name


def _validate_device_preference(preference: str, device_type: str, renderer_name: str) -> None:
    if preference == "dedicated" and device_type != "dedicated":
        raise RuntimeError(
            f"GPU dedicada solicitada pero no disponible. Contexto creado sobre: {renderer_name}"
        )

    if preference == "integrated" and device_type != "integrated":
        raise RuntimeError(
            f"GPU integrada solicitada pero no disponible. Contexto creado sobre: {renderer_name}"
        )


def _create_validated_context(preference: str = "auto"):
    import moderngl

    with _temporary_env(_context_env_overrides(preference)):
        ctx = moderngl.create_standalone_context()

    device_type, renderer_name = _classify_device_info(ctx.info)
    try:
        _validate_device_preference(preference, device_type, renderer_name)
        return ctx, device_type, renderer_name
    except Exception:
        ctx.release()
        raise


def gpu_available(preference: str = "auto") -> bool:
    """Retorna True si ModernGL esta instalado y puede crear un contexto offscreen."""
    try:
        ctx, _, _ = _create_validated_context(preference)
        ctx.release()
        return True
    except Exception:
        return False


class GPURenderer(Renderer):
    """
    Renderer que usa ModernGL para renderizar triangulos en la GPU
    y calcular fitness directamente en shaders.
    """

    def __init__(
        self,
        width: int,
        height: int,
        target_image: np.ndarray,
        device_preference: str = "auto",
    ):
        import moderngl

        self.width = width
        self.height = height
        self.ctx, self.device_type, self.renderer_name = _create_validated_context(device_preference)
        self.device_info = f"{self.device_type}: {self.renderer_name}"

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

        logger.info(
            "GPURenderer inicializado: %dx%d, backend ModernGL, device %s",
            width,
            height,
            self.device_info,
        )

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

        self.fitness_ssim_program = self.ctx.program(
            vertex_shader=fullscreen_vert,
            fragment_shader=_read("fitness_ssim.frag"),
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

        # Pipeline SSIM: textura de bloques (W/8, H/8) + reduccion propia
        self.ssim_block_size = 8
        ssim_w = max(1, self.width // self.ssim_block_size)
        ssim_h = max(1, self.height // self.ssim_block_size)
        self.ssim_texture = self.ctx.texture((ssim_w, ssim_h), 4, dtype='f4')
        self.ssim_fbo = self.ctx.framebuffer(color_attachments=[self.ssim_texture])
        self.ssim_reduction_levels = []
        sw, sh = ssim_w, ssim_h
        while sw > 1 or sh > 1:
            sw, sh = max(1, sw // 2), max(1, sh // 2)
            tex = self.ctx.texture((sw, sh), 4, dtype='f4')
            fbo = self.ctx.framebuffer(color_attachments=[tex])
            self.ssim_reduction_levels.append((tex, fbo, sw, sh))

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
        self.render(genes, gene_type=gene_type)

        if fitness_type == "ssim":
            return self._compute_fitness_ssim()
        return self._compute_fitness_error(fitness_type)

    def _compute_fitness_error(self, fitness_type: str) -> float:
        """Pipeline MSE/MAE: error por pixel → reduccion → 1/(1+error)."""
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

        mean_val = self._reduce(self.error_texture, self.reduction_levels)
        fitness = 1.0 / (1.0 + mean_val)
        return fitness

    def _compute_fitness_ssim(self) -> float:
        """Pipeline SSIM: estadisticas por bloque → reduccion → mean SSIM."""
        ssim_w = max(1, self.width // self.ssim_block_size)
        ssim_h = max(1, self.height // self.ssim_block_size)

        self.ssim_fbo.use()
        self.ctx.viewport = (0, 0, ssim_w, ssim_h)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)

        self.render_texture.use(location=0)
        self.target_texture.use(location=1)
        self.fitness_ssim_program['u_generated'].value = 0
        self.fitness_ssim_program['u_target'].value = 1
        self.fitness_ssim_program['u_image_size'].value = (self.width, self.height)
        self.fitness_ssim_program['u_block_size'].value = self.ssim_block_size

        vao = self.ctx.vertex_array(
            self.fitness_ssim_program,
            [(self.quad_vbo, '2f', 'in_position')],
        )
        vao.render(mode=self.ctx.TRIANGLE_STRIP)
        vao.release()

        mean_ssim = self._reduce(self.ssim_texture, self.ssim_reduction_levels)
        return float(np.clip(mean_ssim, 0.0, 1.0))

    def _reduce(self, source_texture, reduction_levels) -> float:
        """Reduccion iterativa 2x2 hasta 1x1. Retorna media de canales RGB."""
        prev_texture = source_texture
        for tex, fbo, w, h in reduction_levels:
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

        self.ctx.viewport = (0, 0, self.width, self.height)

        _, last_fbo, _, _ = reduction_levels[-1]
        raw = last_fbo.read(components=4, dtype='f4')
        rgba = np.frombuffer(raw, dtype=np.float32)
        return float(np.mean(rgba[:3]))

    @staticmethod
    def detect_device(preference: str = "auto") -> str:
        """Detecta el dispositivo GPU disponible."""
        try:
            ctx, device_type, renderer_name = _create_validated_context(preference)
            ctx.release()
            return f"{device_type}: {renderer_name}"
        except Exception as e:
            logger.warning("No se pudo detectar GPU para preferencia '%s': %s", preference, e)
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
        self.ssim_texture.release()
        self.ssim_fbo.release()
        for tex, fbo, _, _ in self.ssim_reduction_levels:
            tex.release()
            fbo.release()
        self.quad_vbo.release()
        self.triangle_program.release()
        self.fitness_mse_program.release()
        self.fitness_mae_program.release()
        self.fitness_ssim_program.release()
        self.reduce_program.release()
        self.ctx.release()
        logger.info("GPURenderer: recursos liberados")

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
