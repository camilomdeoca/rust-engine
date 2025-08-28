uint pack11_10_11(vec3 rgb)
{
    // Clamp to [0, 1]
    rgb = clamp(rgb, 0.0, 1.0);

    // Scale to bit ranges
    uint r = uint(round(rgb.r * 2047.0)); // 11 bits -> [0, 2047]
    uint g = uint(round(rgb.g * 1023.0)); // 10 bits -> [0, 1023]
    uint b = uint(round(rgb.b * 2047.0)); // 11 bits -> [0, 2047]

    // Pack into uint: [rrrrrrrrrrr gggggggggg bbbbbbbbbbb]
    return (r << 21) | (g << 11) | b;
}

vec3 unpack11_10_11(uint packed)
{
    uint r = (packed >> 21) & 0x7FFu; // 11 bits
    uint g = (packed >> 11) & 0x3FFu; // 10 bits
    uint b = (packed      ) & 0x7FFu; // 11 bits

    return vec3(
        float(r) / 2047.0,
        float(g) / 1023.0,
        float(b) / 2047.0
    );
}

vec2 OctWrap(vec2 v)
{
    return (vec2(1.0) - abs(v.yx)) * vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}
 
vec2 EncodeNormal(vec3 n)
{
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    n.xy = n.z >= 0.0 ? n.xy : OctWrap(n.xy);
    n.xy = n.xy * 0.5 + 0.5;
    return n.xy;
}
 
vec3 DecodeNormal(vec2 f)
{
    f = f * 2.0 - 1.0;
 
    // https://twitter.com/Stubbesaurus/status/937994790553227264
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}
