// Hlsl utilities.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_UTILS_H_
#define _DX11_RENDERER_SHADERS_UTILS_H_

// ---------------------------------------------------------------------------
// Constants.
// ---------------------------------------------------------------------------

static const float PI = 3.14159265358979323846f;
static const float RECIP_PI = 0.31830988618379067153776752674503f;

// ---------------------------------------------------------------------------
// Utility functions.
// ---------------------------------------------------------------------------

float3x3 create_tbn(float3 normal) {
    float3 a0;
    if (abs(normal.x) < abs(normal.y)) {
        const float zup = abs(normal.z) < abs(normal.x) ? 0.0f : 1.0f;
        a0 = float3(zup, 0.0f, 1.0f - zup);
    } else {
        const float zup = (abs(normal.z) < abs(normal.y)) ? 0.0f : 1.0f;
        a0 = float3(0.0f, zup, 1.0f - zup);
    }

    float3 bitangent = normalize(cross(normal, a0));
    float3 tangent = normalize(cross(bitangent, normal));

    float3x3 tbn;
    tbn[0] = tangent;
    tbn[1] = bitangent;
    tbn[2] = normal;
    return tbn;
}

//-----------------------------------------------------------------------------
// Math utils
//-----------------------------------------------------------------------------

float heaviside(float v) {
    return v >= 0.0f ? 1.0f : 0.0f;
}

float pow5(float x) {
    float xx = x * x;
    return xx * xx * x;
}

float schlick_fresnel(float abs_cos_theta) {
    return pow5(1.0f - abs_cos_theta);
}

float schlick_fresnel(float incident_specular, float abs_cos_theta) {
    return incident_specular + (1.0f - incident_specular) * pow5(1.0f - abs_cos_theta);
}

float2 direction_to_latlong_texcoord(float3 direction) {
    float u = (atan2(direction.x, direction.z) + PI) * 0.5f * RECIP_PI; // TODO Turn into a MADD instruction.
    float v = (asin(direction.y) + PI * 0.5f) * RECIP_PI;
    return float2(u, v);
}

#endif // _DX11_RENDERER_SHADERS_UTILS_H_