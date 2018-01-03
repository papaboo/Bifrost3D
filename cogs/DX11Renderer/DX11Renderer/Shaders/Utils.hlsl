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

// ------------------------------------------------------------------------------------------------
// Cone functions.
// ------------------------------------------------------------------------------------------------

// Map a sphere to the spherical cap at origo.
Cone sphere_to_sphere_cap(float3 position, float radius) {
    float sin_theta_sqrd = clamp(radius * radius / dot(position, position), 0.0f, 1.0f);
    return Cone::make(normalize(position), sqrt(1.0f - sin_theta_sqrd));
}

float solidangle(Cone c) { return TWO_PI - TWO_PI * c.cos_theta; }

// Based on Oat and Sander's 2007 technique in Ambient aperture lighting.
float solidangle_of_intersection(Cone c1, Cone c2) {
    float r1 = acos(c1.cos_theta);
    float r2 = acos(c2.cos_theta);
    float rd = acos(dot(c1.direction, c2.direction));

    if (rd <= max(r1, r2) - min(r1, r2)) // TODO replace by abs(r1 - r2)
        // One cap is completely inside the other
        return TWO_PI - TWO_PI * max(c1.cos_theta, c2.cos_theta);
    else if (rd >= r1 + r2)
        // No intersection exists
        return 0.0f;
    else {
        float diff = abs(r1 - r2);
        float den = r1 + r2 - diff;
        float x = 1.0f - (rd - diff) / den; // NOTE Clamped in the original code, but clamping was removed due to smoothstep itself clamping.
        return smoothstep(0.0f, 1.0f, x) * (TWO_PI - TWO_PI * max(c1.cos_theta, c2.cos_theta));
    }
}

// The centroid of the intersection of the two cones.
// See Ambient aperture lighting, 2007, section 3.3.
float3 centroid_of_intersection(Cone c1, Cone c2) {
    float r1 = acos(c1.cos_theta);
    float r2 = acos(c2.cos_theta);
    float d = acos(dot(c1.direction, c2.direction));

    if (d <= max(r1, r2) - min(r1, r2))
        // One cap is completely inside the other
        return c1.cos_theta > c2.cos_theta ? c1.direction : c2.direction;
    else {
        float w = (r2 - r1 + d) / (2.0f * d);
        return normalize(lerp(c2.direction, c1.direction, saturate(w)));
    }
}

#endif // _DX11_RENDERER_SHADERS_UTILS_H_