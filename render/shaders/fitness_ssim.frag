#version 330 core
uniform sampler2D u_generated;
uniform sampler2D u_target;
uniform ivec2 u_image_size;
uniform int u_block_size;
out vec4 fragColor;

void main() {
    ivec2 block_coord = ivec2(gl_FragCoord.xy);
    ivec2 pixel_start = block_coord * u_block_size;
    float n = float(u_block_size * u_block_size);

    // Paso 1: medias
    vec3 sum_g = vec3(0.0), sum_t = vec3(0.0);
    for (int dy = 0; dy < u_block_size; dy++)
        for (int dx = 0; dx < u_block_size; dx++) {
            vec2 uv = (vec2(pixel_start + ivec2(dx, dy)) + 0.5) / vec2(u_image_size);
            sum_g += texture(u_generated, uv).rgb;
            sum_t += texture(u_target, uv).rgb;
        }
    vec3 mu_g = sum_g / n, mu_t = sum_t / n;

    // Paso 2: varianzas y covarianza
    vec3 var_g = vec3(0.0), var_t = vec3(0.0), cov = vec3(0.0);
    for (int dy = 0; dy < u_block_size; dy++)
        for (int dx = 0; dx < u_block_size; dx++) {
            vec2 uv = (vec2(pixel_start + ivec2(dx, dy)) + 0.5) / vec2(u_image_size);
            vec3 g = texture(u_generated, uv).rgb - mu_g;
            vec3 t = texture(u_target, uv).rgb - mu_t;
            var_g += g * g;
            var_t += t * t;
            cov += g * t;
        }
    var_g /= n;
    var_t /= n;
    cov /= n;

    const float C1 = 0.0001;  // 0.01^2
    const float C2 = 0.0009;  // 0.03^2
    vec3 num = (2.0 * mu_g * mu_t + C1) * (2.0 * cov + C2);
    vec3 den = (mu_g * mu_g + mu_t * mu_t + C1) * (var_g + var_t + C2);
    fragColor = vec4(clamp(num / den, 0.0, 1.0), 1.0);
}
