#version 330 core

in vec2 v_local;
in vec4 v_color;
out vec4 fragColor;

void main() {
    // Mantener solo los fragmentos dentro del disco unitario local.
    if (dot(v_local, v_local) > 1.0) {
        discard;
    }
    fragColor = v_color;
}
