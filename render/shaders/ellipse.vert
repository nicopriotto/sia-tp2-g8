#version 330 core

in vec2 in_corner;   // coordenadas locales en [-1, 1]
in vec2 in_center;   // centro de la elipse en [0, 1]
in vec2 in_radii;    // semiejes (rx, ry) en [0, 0.5]
in float in_theta;   // rotacion en radianes
in vec4 in_color;    // RGBA en [0, 1]

out vec2 v_local;
out vec4 v_color;

void main() {
    float c = cos(in_theta);
    float s = sin(in_theta);

    vec2 scaled = in_corner * in_radii;
    vec2 rotated = vec2(
        c * scaled.x - s * scaled.y,
        s * scaled.x + c * scaled.y
    );

    vec2 pos = in_center + rotated;

    // Convertir de [0, 1] a clip space [-1, 1], invirtiendo Y
    float clip_x = pos.x * 2.0 - 1.0;
    float clip_y = 1.0 - pos.y * 2.0;
    gl_Position = vec4(clip_x, clip_y, 0.0, 1.0);

    v_local = in_corner;
    v_color = in_color;
}
