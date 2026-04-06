#version 330 core

uniform sampler2D u_generated;
uniform sampler2D u_target;
out vec4 fragColor;

void main() {
    vec2 uv = gl_FragCoord.xy / textureSize(u_generated, 0);
    vec3 gen = texture(u_generated, uv).rgb;
    vec3 tgt = texture(u_target, uv).rgb;
    vec3 diff = gen - tgt;
    fragColor = vec4(diff * diff, 1.0);
}
