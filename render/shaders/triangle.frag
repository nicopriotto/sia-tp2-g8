#version 330 core

uniform vec4 u_color;  // RGBA en [0, 1]
out vec4 fragColor;

void main() {
    fragColor = u_color;
}
