// OptiX shading utilities.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_UTILS_H_
#define _OPTIXRENDERER_SHADING_UTILS_H_

#include <OptiXRenderer/Defines.h>

#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include <cmath>

namespace OptiXRenderer {

//-----------------------------------------------------------------------------
// Constants
//-----------------------------------------------------------------------------

__constant_all__ float COAT_SPECULARITY = 0.04f;
__constant_all__ float COAT_IOR = 1.5f;
__constant_all__ float AIR_IOR = 1.0f;

//-----------------------------------------------------------------------------
// Math utils
//-----------------------------------------------------------------------------

__inline_all__ float average(optix::float3 v) {
    return (v.x + v.y + v.z) / 3.0f;
}

// Toroidal shift of a base random number.
__inline_all__ optix::float4 toroidal_shift(optix::float4 base, optix::float4 shift) {
    optix::float4 s = base + shift;
    return s - floor(s);
}

__inline_all__ float heaviside(float v) {
    return v >= 0.0f ? 1.0f : 0.0f;
}

__inline_all__ bool same_hemisphere(optix::float3 wo, optix::float3 wi) {
    return wo.z * wi.z >= 0.0f;
}

// Fix shading normals that are facing away from the camera that intersected the surface.
// The issue happens when a ray intersects a triangle, where the shading normal at
// one of more vertices are pointing away from the view direction. Think tesselated sphere.
// We 'fix' this by offsetting the shading normal along the view direction, w, until it is no longer viewed from behind.
// w is assumed to be normalized.
// It's possible to set a target cos(angle) or dot(w,n). As the output normal is normalized afterwards,
// the target is going to be overshot by by the returned normal,
// but as it's generally smaller cos_theta adjustments we're interested in that's acceptable.
__inline_all__ optix::float3 fix_backfacing_shading_normal(optix::float3 w, optix::float3 n, float target_cos_theta = 0.0f) {
    float cos_theta = optix::dot(w, n);
    if (cos_theta < target_cos_theta) {
        float c = cos_theta - target_cos_theta;
        return optix::normalize(n - c * w);
    } else
        return n;
}

__inline_all__ float sign(float v) {
    return v >= 0.0f ? 1.0f : -1.0f;
}

__inline_all__ float sum(optix::float3 v) {
    return v.x + v.y + v.z;
}

__inline_all__ float length_squared(optix::float3 v) {
    return optix::dot(v, v);
}

__inline_all__ optix::double3 lerp_double(const optix::double3& a, const optix::double3& b, const double t) {
    return optix::make_double3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t);
}

__inline_all__ bool is_black(optix::float3 color) {
    return color.x <= 0.0f && color.y <= 0.0f && color.z <= 0.0f;
}

__inline_all__ float saturate(float v) {
    return optix::clamp(v, 0.0f, 1.0f);
}

__inline_all__ float pow2(float x) {
    return x * x;
}

__inline_all__ optix::float3 pow2(optix::float3 x) {
    return x * x;
}

__inline_all__ float pow4(float x) {
    float xx = x * x;
    return xx * xx;
}

__inline_all__ float pow5(float x) {
    float xx = x * x;
    return xx * xx * x;
}

// Specularity of dielectrics at normal incidence, where the ray is leaving a medium with index of refraction ior_o
// and entering a medium with index of refraction, ior_i.
// Ray Tracing Gems 2, Chapter 9, The Schlick Fresnel Approximation, page 110 footnote.
__inline_all__ float dielectric_specularity(float ior_o, float ior_i) {
    return pow2((ior_o - ior_i) / (ior_o + ior_i));
}

// Specularity of dielectrics at normal incidence, where the ray is leaving a dielectric medium with index of refraction ior_o
// and entering a conductor medium with index of refraction, ior_i, and extinction coefficient, ext_i.
__inline_all__ optix::float3 conductor_specularity(optix::float3 ior_o, optix::float3 ior_i, optix::float3 ext_i) {
    optix::float3 ext_i_sqrd = pow2(ext_i);
    return (pow2(ior_o - ior_i) + ext_i_sqrd) / (pow2(ior_o + ior_i) + ext_i_sqrd);
}

// Estimates a dielectric's index of refraction from specularity.
// It is assumed that the specularity describes the specularity of the material when bordering air, i.e ior_o is 1.0.
// Finding the index of refraction requires solving a second degree polynomial with two solutions.
// For dielectrics the solution with the largest value is the correct one.
// The whole thing can be reduced to the expression below.
// Source: Extending the Disney BRDF to a BSDF with Integrated Subsurface Scattering, section 3.2, Burley, 2015
__inline_all__ float dielectric_ior_from_specularity(float specularity) {
    return 2.0f / (1.0f - sqrt(specularity)) - 1.0f;
}

// Estimates a conductor's index of refraction from specularity.
// It is assumed that the specularity describes the specularity of the material when bordering air, i.e ior_o is 1.0.
// Finding the index of refraction requires solving a second degree polynomial with two solutions.
// For dielectrics the solution with the lowest value is the correct one.
__inline_all__ optix::float3 conductor_ior_from_specularity(optix::float3 specularity, optix::float3 ext_i) {
    optix::float3 a = specularity - 1;
    optix::float3 b = 2 * specularity + 2;
    optix::float3 c = a + (specularity - 1) * pow2(ext_i);
    optix::float3 d = b * b - 4 * a * c;
    optix::float3 sqrt_d = { sqrt(d.x), sqrt(d.y), sqrt(d.z) };
    return (-b + sqrt_d) / (2 * a);
}

// Adjust the specularity of a dielectric material, which is set with the assumption that the material is seen through air,
// to the specularity that the material would have as seen through a volume with the ior defined by the exterior ior.
__inline_all__ float adjust_dielectric_specularity_to_exterior_medium(float exterior_ior, float specularity_through_air) {
    // Convert specularity to base_ior
    float base_ior = dielectric_ior_from_specularity(specularity_through_air);

    // Compute new base specularity
    return dielectric_specularity(exterior_ior, base_ior);
}

// Adjust the specularity of a conductor material, which is set with the assumption that the material is seen through air,
// to the specularity that the material would have as seen through a volume with the ior defined by the exterior ior.
__inline_all__ optix::float3 adjust_conductor_specularity_to_exterior_medium(optix::float3 exterior_ior, optix::float3 specularity_through_air, optix::float3 extinction_coefficient) {
    // Convert specularity to base_ior
    optix::float3 base_ior = conductor_ior_from_specularity(specularity_through_air, extinction_coefficient);

    // Compute new base specularity
    return conductor_specularity(exterior_ior, base_ior, extinction_coefficient);
}

__inline_all__ float schlick_fresnel(float incident_specular, float abs_cos_theta) {
    return incident_specular + (1.0f - incident_specular) * pow5(1.0f - abs_cos_theta);
}

__inline_all__ optix::float3 schlick_fresnel(optix::float3 incident_specular, float abs_cos_theta) {
    float t = pow5(1.0f - abs_cos_theta);
    return (1.0f - t) * incident_specular + t;
}

__inline_all__ float dielectric_schlick_fresnel(float incident_specular, float abs_cos_theta, float ior_i_over_o) {
    // Return 1.0 for full reflection in case of total internal reflection.
    // cos(theta) is expected to be absolute and ior_i_over_o to have been preadjusted to fit the side hit.
    // Sources:
    // * PBRT's FrDielectric in scattering.h
    // * https://raytracing.github.io/books/RayTracingInOneWeekend.html#dielectrics/totalinternalreflection
    float sin2_theta = 1 - pow2(abs_cos_theta);
    if (sin2_theta >= pow2(ior_i_over_o))
        return 1.0f;

    float t = pow5(1.0f - abs_cos_theta);
    return (1.0f - t) * incident_specular + t;
}

//-----------------------------------------------------------------------------
// Trigonometri utils
//-----------------------------------------------------------------------------

__inline_all__ void sincos(float theta, float& sin_theta, float& cos_theta) {
#if GPU_DEVICE
    sincosf(theta, &sin_theta, &cos_theta);
#else
    sin_theta = sin(theta);
    cos_theta = cos(theta);
#endif
}

__inline_all__ float cos_theta(optix::float3 w) { return w.z; }
__inline_all__ float cos2_theta(optix::float3 w) { return w.z * w.z; }
__inline_all__ float abs_cos_theta(optix::float3 w) { return abs(w.z); }

__inline_all__ float sin2_theta(optix::float3 w) { return fmaxf(0.0f, 1.0f - cos2_theta(w)); }
__inline_all__ float sin_theta(optix::float3 w) { return sqrt(sin2_theta(w)); }

__inline_all__ float tan_theta(optix::float3 w) { return sin_theta(w) / cos_theta(w); }
__inline_all__ float tan2_theta(optix::float3 w) { return sin2_theta(w) / cos2_theta(w); }

__inline_all__ float cos_phi(optix::float3 w) {
    float sin_theta_ = sin_theta(w);
    return (sin_theta_ == 0) ? 1.0f : optix::clamp(w.x / sin_theta_, -1.0f, 1.0f);
}
__inline_all__ float cos2_phi(optix::float3 w) { return pow2(cos_phi(w)); }

__inline_all__ float sin_phi(optix::float3 w) {
    float sin_theta_ = sin_theta(w);
    return (sin_theta_ == 0) ? 0.0f : optix::clamp(w.y / sin_theta_, -1.0f, 1.0f);
}
__inline_all__ float sin2_phi(optix::float3 w) { return pow2(sin_phi(w)); }

//-----------------------------------------------------------------------------
// Utility functions
//-----------------------------------------------------------------------------

__inline_all__ optix::float4 unorm8_to_float(optix::uchar4 unorms) {
    constexpr float s = 1 / 255.0f;
    return optix::make_float4(unorms.x, unorms.y, unorms.z, unorms.w) * s;
}

__inline_all__ optix::uchar4 float_to_unorm8(optix::float4 values) {
    return optix::make_uchar4(unsigned char(saturate(values.x) * 255.0f + 0.5f), unsigned char(saturate(values.y) * 255.0f + 0.5f),
                              unsigned char(saturate(values.z) * 255.0f + 0.5f), unsigned char(saturate(values.w) * 255.0f + 0.5f));
}

__inline_all__ optix::float2 direction_to_latlong_texcoord(optix::float3 direction) {
    float u = (atan2f(direction.z, direction.x) + PIf) * 0.5f / PIf;
    float v = (asinf(direction.y) + PIf * 0.5f) / PIf;
    return optix::make_float2(u, v);
}

__inline_all__ optix::float3 latlong_texcoord_to_direction(optix::float2 uv) {
    float phi = uv.x * 2.0f * PIf;
    float theta = uv.y * PIf;
    float sin_theta, cos_theta, sin_phi, cos_phi;
    sincos(theta, sin_theta, cos_theta);
    sincos(phi, sin_phi, cos_phi);
    return -optix::make_float3(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
}

// Insert a 0 bit in between each of the 16 low bits of v.
__inline_all__ unsigned int part_by_1(unsigned int v) {
    v &= 0x0000ffff;                 // v = ---- ---- ---- ---- fedc ba98 7654 3210
    v = (v ^ (v << 8)) & 0x00ff00ff; // v = ---- ---- fedc ba98 ---- ---- 7654 3210
    v = (v ^ (v << 4)) & 0x0f0f0f0f; // v = ---- fedc ---- ba98 ---- 7654 ---- 3210
    v = (v ^ (v << 2)) & 0x33333333; // v = --fe --dc --ba --98 --76 --54 --32 --10
    v = (v ^ (v << 1)) & 0x55555555; // v = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return v;
}

__inline_all__ unsigned int morton_encode(unsigned int x, unsigned int y) {
    return part_by_1(y) | (part_by_1(x) << 1);
}

// Delete all bits not at positions divisible by 3 and compacts the rest.
__inline_all__ unsigned int compact_by_2(unsigned int v) {
    v &= 0x09249249;                  // v = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    v = (v ^ (v >> 2)) & 0x030c30c3;  // v = ---- --98 ---- 76-- --54 ---- 32-- --10
    v = (v ^ (v >> 4)) & 0x0300f00f;  // v = ---- --98 ---- ---- 7654 ---- ---- 3210
    v = (v ^ (v >> 8)) & 0xff0000ff;  // v = ---- --98 ---- ---- ---- ---- 7654 3210
    v = (v ^ (v >> 16)) & 0x000003ff; // v = ---- ---- ---- ---- ---- --98 7654 3210
    return v;
}

__inline_all__ optix::uint3 morton_decode_3D(unsigned int v) {
    return optix::make_uint3(compact_by_2(v >> 2), compact_by_2(v >> 1), compact_by_2(v));
}

__inline_all__ unsigned int reverse_bits(unsigned int n) {
#if GPU_DEVICE
    n = __brev(n);
#else
    n = (n << 16) | (n >> 16);
    n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
    n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
    n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
    n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
#endif
    return n;
}

// Computes a tangent and bitangent that together with the normal creates an orthonormal basis.
// Building an Orthonormal Basis, Revisited, Duff et al.
// http://jcgt.org/published/0006/01/01/paper.pdf
__inline_all__ void compute_tangents(optix::float3 normal,
    optix::float3& tangent, optix::float3& bitangent) {
    using namespace optix;

    float sign = copysignf(1.0f, normal.z);
    const float a = -1.0f / (sign + normal.z);
    const float b = normal.x * normal.y * a;
    tangent = { 1.0f + sign * normal.x * normal.x * a, sign * b, -sign * normal.x };
    bitangent = { b, sign + normal.y * normal.y * a, -normal.y };
}

// Scales the roughness of a material placed underneath a rough coat layer.
// This is done to simulate how a wider lobe from the rough transmission would
// perceptually widen the specular lobe of the underlying material.
// The implementation is based on equation 86 in the Roughening chapter of the OpenPBR course notes for Physically Based Shading 2025.
// https://blog.selfshadow.com/publications/s2025-shading-course/
__inline_all__ float modulate_roughness_under_coat(float base_roughness, float coat_roughness) {
    float x_coat = 1 - AIR_IOR / COAT_IOR;
    float adjusted_roughness4 = fminf(1, pow4(base_roughness) + 2.0f * x_coat * pow4(coat_roughness));
    return pow(adjusted_roughness4, 0.25f);
}

// Offset ray origin along the geometric normal. Values should ideally be in world space.
// Source: Ray Tracing Gems 1, Chapter 6, A Fast and Robust Method for Avoiding Self-Intersection
// For a more stable approach see https://developer.nvidia.com/blog/solving-self-intersection-artifacts-in-directx-raytracing/
__inline_all__ optix::float3 offset_ray_origin(optix::float3 ray_world_origin, optix::float3 world_normal) {
    using namespace optix;

    constexpr float origin = 1.0f / 32.0f;
    constexpr float float_scale = 1.0f / 65536.0f;
    constexpr float int_scale = 256.0f;

    int3 of_i = make_int3(int_scale * world_normal);

    float3 p_i = make_float3(int_as_float(float_as_int(ray_world_origin.x) + ((ray_world_origin.x < 0) ? -of_i.x : of_i.x)),
                             int_as_float(float_as_int(ray_world_origin.y) + ((ray_world_origin.y < 0) ? -of_i.y : of_i.y)),
                             int_as_float(float_as_int(ray_world_origin.z) + ((ray_world_origin.z < 0) ? -of_i.z : of_i.z)));

    return make_float3(fabsf(ray_world_origin.x) < origin ? ray_world_origin.x + float_scale * world_normal.x : p_i.x,
                       fabsf(ray_world_origin.y) < origin ? ray_world_origin.y + float_scale * world_normal.y : p_i.y,
                       fabsf(ray_world_origin.z) < origin ? ray_world_origin.z + float_scale * world_normal.z : p_i.z);
}

// Offset ray origin along the geometric normal in the direction of the ray direction.
// Values should ideally be in world space for best precision.
__inline_all__ optix::float3 offset_ray_origin(optix::float3 ray_origin, optix::float3 ray_direction,
                                               optix::float3 geometric_normal) {
    float cos_theta = optix::dot(geometric_normal, ray_direction);
    geometric_normal = cos_theta >= 0 ? geometric_normal : -geometric_normal;
    return offset_ray_origin(ray_origin, geometric_normal);
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_UTILS_H_