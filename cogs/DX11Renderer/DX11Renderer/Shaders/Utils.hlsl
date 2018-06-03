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

struct TextureBound {
    static const unsigned int None = 0;
    static const unsigned int Tint = 1 << 0;
    static const unsigned int Coverage = 1 << 1;
};

struct SceneVariables {
    float4x4 view_projection_matrix;
    float4 camera_position;
    float4 environment_tint; // .w component is 1 if an environment tex is bound, otherwise 0.
    float4x4 inverted_view_projection_matrix;
    float4x4 projection_matrix;
    float4x4 inverted_projection_matrix;
    float4x3 world_to_view_matrix;
};

// ------------------------------------------------------------------------------------------------
// Math utils
// ------------------------------------------------------------------------------------------------

unsigned int ceil_divide(unsigned int a, unsigned int b) {
    return (a / b) + ((a % b) > 0);
}

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

float luminance(float3 color) {
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f));
}

inline float inverse_lerp(const float a, const float b, const float v) {
    return (v - a) / (b - a);
}

float reciprocal_length(float3 v) {
    return rsqrt(length_squared(v));
}

float schlick_fresnel(float abs_cos_theta) {
    return pow5(1.0f - abs_cos_theta);
}

float schlick_fresnel(float incident_specular, float abs_cos_theta) {
    return incident_specular + (1.0f - incident_specular) * pow5(1.0f - abs_cos_theta);
}

float sign(float v) { return v >= 0.0f ? +1.0f : -1.0f; }

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

float3 decode_octahedral_normal(int packed_encoded_normal) {
    const int SHRT_MAX = 32767;
    const int SHRT_MIN = -32768;

    // Unpack the 2 shorts representing the encoded normal. 
    // The sign is implecitly handled for the 16 most significant bits, but needs to be explicitly handled for the least ones.
    int encoding_x = (packed_encoded_normal & 0xFFFF) + SHRT_MIN;
    int encoding_y = packed_encoded_normal >> 16;

    float2 p2 = float2(encoding_x, encoding_y);
    float3 n = float3(p2, SHRT_MAX - abs(p2.x) - abs(p2.y));
    if (n.z < 0.0f) {
        float tmp_x = (SHRT_MAX - abs(n.y)) * sign(n.x);
        n.y = (SHRT_MAX - abs(n.x)) * sign(n.y);
        n.x = tmp_x;
    }
    return n;
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

    if (rd <= abs(r2 - r1))
        // One cap is completely inside the other
        return TWO_PI - TWO_PI * max(c1.cos_theta, c2.cos_theta);
    else if (rd >= r1 + r2)
        // No intersection exists
        return 0.0f;
    else {
        float diff = abs(r2 - r1);
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

    if (d <= abs(r2 - r1))
        // One cap is completely inside the other
        return c1.cos_theta > c2.cos_theta ? c1.direction : c2.direction;
    else {
        float w = (r2 - r1 + d) / (2.0f * d);
        return normalize(lerp(c2.direction, c1.direction, saturate(w)));
    }
}

struct  CentroidAndSolidangle {
    float3 centroid_direction;
    float solidangle;
};

// Computes the centroid and solidangle of the intersection from the cone with the hemisphere.
// Assumes that the cone has a maximum angle of 90 degrees (positive cos theta).
CentroidAndSolidangle centroid_and_solidangle_on_hemisphere(Cone cone) {
    const Cone hemipshere = { float3(0.0f, 0.0f, 1.0f), 0.0f };

    float r1 = acos(cone.cos_theta);
    float r2 = 1.57079637f;
    float rd = acos(cone.direction.z);

    if (rd <= r2 - r1) {
        // One cone is completely inside the other
        float3 centroid_direction = cone.cos_theta > hemipshere.cos_theta ? cone.direction : hemipshere.direction;
        float solidangle = TWO_PI - TWO_PI * cone.cos_theta;
        return Cone::make(centroid_direction, solidangle);
    }
    else {
        float w = (r2 - r1 + rd) / (2.0f * rd);
        float3 centroid_direction = normalize(lerp(hemipshere.direction, cone.direction, w));

        if (rd >= r1 + r2)
            // No intersection exists
            return Cone::make(centroid_direction, 0.0f);
        else {
            float diff = r2 - r1;
            float den = 2.0f * r1;
            float x = 1.0f - (rd - diff) / den;
            float solidangle = smoothstep(0.0f, 1.0f, x) * (TWO_PI - TWO_PI * cone.cos_theta);
            return Cone::make(centroid_direction, solidangle);
        }
    }
}

#endif // _DX11_RENDERER_SHADERS_UTILS_H_