// Hlsl utilities.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_UTILS_H_
#define _DX11_RENDERER_SHADERS_UTILS_H_

// ------------------------------------------------------------------------------------------------
// Renderer wide resources.
// ------------------------------------------------------------------------------------------------

SamplerState point_sampler : register(s14);
SamplerState bilinear_sampler : register(s15);

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

// Mirrored CPU side by Renderer::SceneConstants
struct SceneVariables {
    float4x4 view_projection_matrix;
    float4 environment_tint; // .w component is 1 if an environment tex is bound, otherwise 0.
    int2 g_buffer_to_ao_index_offset;
    int2 viewport_size;
    float4x4 inverted_view_projection_matrix;
    float4x4 projection_matrix;
    float4x4 inverted_projection_matrix;
    float4x3 world_to_view_matrix;
    float4x3 view_to_world_matrix;

    float3 camera_world_position() { return view_to_world_matrix._m30_m31_m32; }
};

// ------------------------------------------------------------------------------------------------
// Math utils
// ------------------------------------------------------------------------------------------------

unsigned int ceil_divide(unsigned int a, unsigned int b) { return (a / b) + ((a % b) > 0); }

float heaviside(float v) { return v >= 0.0f ? 1.0f : 0.0f; }

float non_zero_sign(float v) { return v >= 0.0f ? +1.0f : -1.0f; }
float2 non_zero_sign(float2 v) { return float2(non_zero_sign(v.x), non_zero_sign(v.y)); }

float pow2(float x) { return x * x; }
float pow4(float x) { float xx = x * x; return xx * xx; }
float pow5(float x) { return pow4(x) * x; }

float length_squared(float3 v) { return dot(v, v); }

float luminance(float3 color) { return dot(color, float3(0.2126f, 0.7152f, 0.0722f)); }

float inverse_lerp(float a, float b, float v) { return (v - a) / (b - a); }

float reciprocal_length(float3 v) { return rsqrt(length_squared(v)); }

float schlick_fresnel(float abs_cos_theta) { return pow5(1.0 - abs_cos_theta); }
float schlick_fresnel(float incident_specular, float abs_cos_theta) { 
    return incident_specular + (1.0 - incident_specular) * pow5(1.0 - abs_cos_theta);
}
float3 schlick_fresnel(float3 incident_specular, float abs_cos_theta) { 
    float t = pow5(1.0 - abs_cos_theta);
    return (1 - t) * incident_specular + t;
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

// Fix shading normals that are facing away from the camera that intersected the surface.
// The issue happens when a ray intersects a triangle, where the shading normal at
// one of more vertices are pointing away from the view direction. Think tesselated sphere.
// We 'fix' this by offsetting the shading normal along the view direction, w, until it is no longer viewed from behind.
// w is assumed to be normalized.
// It's possible to set a target cos(angle) or dot(w,n). As the output normal is normalized afterwards,
// the target is going to be overshot by by the returned normal,
// but as it's generally smaller cos_theta adjustments we're interested in that's acceptable.
float3 fix_backfacing_shading_normal(float3 w, float3 n, float target_cos_theta) {
    float cos_theta = dot(w, n);
    if (cos_theta < target_cos_theta) {
        float c = cos_theta - target_cos_theta;
        return normalize(n - c * w);
    } else
        return n;
}

// ------------------------------------------------------------------------------------------------
// Space transformations.
// ------------------------------------------------------------------------------------------------

// Transform depth to view-space position.
// https://mynameismjp.wordpress.com/2009/03/10/reconstructing-position-from-depth/
float3 position_from_depth(float z_over_w, float2 viewport_uv, float4x4 inverted_projection_matrix) {
    // Get x/w and y/w from the viewport position
    float x_over_w = viewport_uv.x * 2 - 1;
    float y_over_w = 1 - 2 * viewport_uv.y;
    float4 projected_position = float4(x_over_w, y_over_w, z_over_w, 1.0f);
    // Transform by the inverse projection matrix
    float4 projected_view_pos = mul(projected_position, inverted_projection_matrix);
    // Divide by w to get the view-space position
    return projected_view_pos.xyz / projected_view_pos.w;
}

// Transform depth to view-space position using a perspective projection matrix.
// https://mynameismjp.wordpress.com/2009/03/10/reconstructing-position-from-depth/
float3 perspective_position_from_depth(float z_over_w, float2 viewport_uv, float4x4 inverted_projection_matrix) {
    // Get x/w and y/w from the viewport position
    float x_over_w = viewport_uv.x * 2 - 1;
    float y_over_w = 1 - 2 * viewport_uv.y;

    // Transform by the inverse projection matrix
    float4 projected_view_pos = { x_over_w * inverted_projection_matrix._m00,
                                  y_over_w * inverted_projection_matrix._m11,
                                  1.0,
                                  z_over_w * inverted_projection_matrix._m23 + inverted_projection_matrix._m33 };
    // Divide by w to get the view-space position
    return projected_view_pos.xyz / projected_view_pos.w;
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

// Scales the roughness of a material placed underneath a rough coat layer.
// This is done simulate how a wider lobe from the rough transmission would
// perceptually widen the specular lobe of the underlying material.
float modulate_roughness_under_coat(float base_roughness, float coat_roughness) {
    return sqrt(1.0f - (1.0f - pow2(base_roughness)) * (1.0f - pow2(coat_roughness)));
}

float3 decode_octahedral_normal(int packed_encoded_normal) {
    const int SHORT_MAX = 32767;
    const int SHORT_MIN = -32768;

    // Unpack the 2 shorts representing the encoded normal. 
    // The sign is implicitly handled for the 16 most significant bits, but needs to be explicitly handled for the least ones.
    int encoding_x = (packed_encoded_normal & 0xFFFF) + SHORT_MIN;
    int encoding_y = packed_encoded_normal >> 16;

    float2 p2 = float2(encoding_x, encoding_y);
    float3 n = float3(p2, SHORT_MAX - abs(p2.x) - abs(p2.y));
    if (n.z < 0.0f) {
        float tmp_x = (SHORT_MAX - abs(n.y)) * non_zero_sign(n.x);
        n.y = (SHORT_MAX - abs(n.x)) * non_zero_sign(n.y);
        n.x = tmp_x;
    }
    return n;
}

float2 encode_octahedral_normal(float3 normal) {
    float l1norm = abs(normal.x) + abs(normal.y) + abs(normal.z);
    float2 result = normal.xy / l1norm;
    if (normal.z < 0.0)
        result = (1.0 - abs(result.yx)) * non_zero_sign(result.xy);
    return result;
}

float3 decode_octahedral_normal(float2 encoded_normal) {
    float3 n = float3(encoded_normal.xy, 1.0 - abs(encoded_normal.x) - abs(encoded_normal.y));
    if (n.z < 0.0)
        n.xy = (1.0 - abs(n.yx)) * non_zero_sign(n.xy);
    return normalize(n);
}

float2 encode_ss_octahedral_normal(float3 normal) {
    float l1norm = abs(normal.x) + abs(normal.y) + abs(normal.z);
    return normal.xy / l1norm;
}

float3 decode_ss_octahedral_normal(float2 encoded_normal) {
    float3 n = float3(encoded_normal.xy, -1.0 + abs(encoded_normal.x) + abs(encoded_normal.y));
    return normalize(n);
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

struct CentroidAndSolidangle {
    float3 centroid_direction;
    float solidangle;
};

// Computes the centroid and solidangle of the intersection from the cone with the hemisphere.
// Assumes that the cone has a maximum angle of 90 degrees (positive cos theta).
CentroidAndSolidangle centroid_and_solidangle_on_hemisphere(Cone cone) {
    const Cone hemisphere = { float3(0.0f, 0.0f, 1.0f), 0.0f };

    float r1 = acos(cone.cos_theta);
    float r2 = 1.57079637f;
    float rd = acos(cone.direction.z);

    if (rd <= r2 - r1) {
        // One cone is completely inside the other
        float3 centroid_direction = cone.cos_theta > hemisphere.cos_theta ? cone.direction : hemisphere.direction;
        float solidangle = TWO_PI - TWO_PI * cone.cos_theta;
        return Cone::make(centroid_direction, solidangle);
    }
    else {
        float w = (r2 - r1 + rd) / (2.0f * rd);
        float3 centroid_direction = normalize(lerp(hemisphere.direction, cone.direction, w));

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