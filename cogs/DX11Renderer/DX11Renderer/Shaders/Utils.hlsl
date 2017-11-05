// Hlsl utilities.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_UTILS_H_
#define _DX11_RENDERER_SHADERS_UTILS_H_

// ------------------------------------------------------------------------------------------------
// Constants.
// ------------------------------------------------------------------------------------------------

static const float PI = 3.14159265358979323846f;
static const float TWO_PI = 6.283185307f;
static const float RECIP_PI = 0.31830988618379067153776752674503f;

// ------------------------------------------------------------------------------------------------
// Types.
// ------------------------------------------------------------------------------------------------

struct Cone {
    float3 direction;
    float cos_theta;

    static Cone make(float3 direction, float cos_theta) {
        Cone cone = { direction, cos_theta };
        return cone;
    }
};

struct Sphere {
    float3 position;
    float radius;

    static Sphere make(float3 position, float radius) {
        Sphere sphere = { position, radius };
        return sphere;
    }
};

// ------------------------------------------------------------------------------------------------
// Utility functions.
// ------------------------------------------------------------------------------------------------

// Computes a tangent and bitangent that together with the normal creates an orthonormal bases.
// Building an Orthonormal Basis, Revisited, Duff et al.
// http://jcgt.org/published/0006/01/01/paper.pdf
float3x3 create_TBN(float3 normal) {
    float sign = normal.z < 0.0f ? -1.0f : 1.0f;
    const float a = -1.0f / (sign + normal.z);
    const float b = normal.x * normal.y * a;
    float3 tangent = { 1.0f + sign * normal.x * normal.x * a, sign * b, -sign * normal.x };
    float3 bitangent = { b, sign + normal.y * normal.y * a, -normal.y };

    float3x3 tbn;
    tbn[0] = tangent;
    tbn[1] = bitangent;
    tbn[2] = normal;
    return tbn;
}

float3x3 create_inverse_TBN(float3 normal) {
    return transpose(create_TBN(normal));
}

// ------------------------------------------------------------------------------------------------
// Math utils
// ------------------------------------------------------------------------------------------------

float heaviside(float v) {
    return v >= 0.0f ? 1.0f : 0.0f;
}

float pow2(float x) {
    return x * x;
}

float pow5(float x) {
    float xx = x * x;
    return xx * xx * x;
}

float length_squared(float3 v) {
    return dot(v, v);
}

float schlick_fresnel(float abs_cos_theta) {
    return pow5(1.0f - abs_cos_theta);
}

float schlick_fresnel(float incident_specular, float abs_cos_theta) {
    return incident_specular + (1.0f - incident_specular) * pow5(1.0f - abs_cos_theta);
}

float2 direction_to_latlong_texcoord(float3 direction) {
    float u = (atan2(direction.z, direction.x) + PI) * 0.5f * RECIP_PI;
    float v = asin(direction.y) * RECIP_PI + 0.5f;
    return float2(u, v);
}

float3 latlong_texcoord_to_direction(float2 uv) {
    float phi = uv.x * 2.0f * PI;
    float theta = uv.y * PI;
    float sin_theta = sin(theta);
    return -float3(sin_theta * cos(phi), cos(theta), sin_theta * sin(phi));
}

#endif // _DX11_RENDERER_SHADERS_UTILS_H_