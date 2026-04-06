#version 330 core

in vec2 in_position;  // coordenadas en [0, 1]
in vec4 in_color;     // RGBA en [0, 1]

out vec4 v_color;

void main() {
    // Convertir de [0, 1] a clip space [-1, 1], invirtiendo Y
    // (imagen: y=0 arriba, OpenGL: y=0 abajo)
    float clip_x = in_position.x * 2.0 - 1.0;
    float clip_y = 1.0 - in_position.y * 2.0;
    gl_Position = vec4(clip_x, clip_y, 0.0, 1.0);
    v_color = in_color;
}
