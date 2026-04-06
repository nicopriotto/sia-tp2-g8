#version 330 core

uniform sampler2D u_input;
uniform vec2 u_texel_size;  // 1.0 / tamano de la textura de entrada
out vec4 fragColor;

void main() {
    // Cada pixel del output promedia un bloque de 2x2 del input
    vec2 uv = gl_FragCoord.xy * 2.0 * u_texel_size;

    vec4 a = texture(u_input, uv);
    vec4 b = texture(u_input, uv + vec2(u_texel_size.x, 0.0));
    vec4 c = texture(u_input, uv + vec2(0.0, u_texel_size.y));
    vec4 d = texture(u_input, uv + u_texel_size);

    fragColor = (a + b + c + d) * 0.25;
}
