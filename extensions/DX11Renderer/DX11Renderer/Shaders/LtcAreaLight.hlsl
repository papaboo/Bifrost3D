// Linear Transformed Cosine area light source approximations.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include "Utils.hlsl"

#ifndef _DX11_RENDERER_SHADERS_LTC_AREA_LIGHT_H_
#define _DX11_RENDERER_SHADERS_LTC_AREA_LIGHT_H_

namespace LtcAreaLight {

float3x3 ltc_shading_space(float3 wo, float3 normal) {
    float wo_dot_n = dot(wo, normal);
    if (abs(wo_dot_n) <= 0.999999) {
        // Construct orthonormal basis around normal with tangent pointing along the view direction.
        float3 tangent = normalize(wo - normal * wo_dot_n);
        float3 bitangent = cross(normal, tangent);
        return float3x3(tangent, bitangent, normal);
    } else
        // Edge case where wo is so similar to the normal that the forward direction is unstable
        // and we just compute some tangent space around the normal.
        return create_TBN(normal);
}

float3 intersection_at_horizon(float3 v0, float3 v1) {
    float t = -v0.z / (v1.z - v0.z);
    return float3(lerp(v0.x, v1.x, t), lerp(v0.y, v1.y, t), 0.0);
}

// Clips the triangle against the horizon. Returns the number of valid vertices returned in the input array.
// If one vertex is below the horizon, then clipping returns a four vertex polygon and the fourth vertex is populated.
// If all vertices are below the horizon then the 0 is returned.
int clip_triangle_to_horizon(inout float3 vertices[4]) {
    // Detect clipping config
    int config = 0;
    if (vertices[0].z > 0) config += 1;
    if (vertices[1].z > 0) config += 2;
    if (vertices[2].z > 0) config += 4;

    // No clipping early outs.
    if (config == 7)
        // Full visibility
        return 3;
    if (config == 0)
        // Triangle is completely below the horizon.
        return 0;

    float3 v0_to_v1 = intersection_at_horizon(vertices[0], vertices[1]);
    float3 v0_to_v2 = intersection_at_horizon(vertices[0], vertices[2]);
    float3 v1_to_v2 = intersection_at_horizon(vertices[1], vertices[2]);

    // Clip triangle
    if (config == 1) {
        // Vertex 0 is above the horizon.
        // Vertex 1 and 2 is projected towards vertex 0.
        vertices[1] = v0_to_v1;
        vertices[2] = v0_to_v2;
        return 3;
    } else if (config == 2) {
        // Vertex 1 is above the horizon.
        // Vertex 0 and 2 is projected towards vertex 1.
        vertices[0] = v0_to_v1;
        vertices[2] = v1_to_v2;
        return 3;
    } else if (config == 3) {
        // Vertex 2 is below, vertex 0 and 1 are above.
        // Project 2 along 1-2 direction and add vertex 4 at 2-0 direction
        vertices[3] = v0_to_v2;
        vertices[2] = v1_to_v2;
        return 4;
    } else if (config == 4) {
        // Vertex 2 is above the horizon.
        // Vertex 0 and 1 is projected towards vertex 2.
        vertices[0] = v0_to_v2;
        vertices[1] = v1_to_v2;
        return 3;
    } else if (config == 5) {
        // Vertex 1 is below, vertex 0 and 2 are above.
        // Copy vertex 0 to vertex 3, to preserve the end winding order, Project 1 along 0-1 direction and add vertex 4 at 2-1 direction
        vertices[3] = vertices[0];
        vertices[0] = v0_to_v1;
        vertices[1] = v1_to_v2;
        return 4;
    } else if (config == 6) {
        // Vertex 0 is below, vertex 1 and 2 are above.
        // Project 0 along 0-1 direction and add vertex 4 at 2-0 direction
        vertices[3] = v0_to_v2;
        vertices[0] = v0_to_v1;
        return 4;
    }

    // Impossible to reach.
    return 0;
}

// Edge integral using the fitted function to replace acos and gain increased precision.
// Real-Time Area Lighting: a Journey from Research to Production, Stephen Hill and Eric Heitz, Siggraph, 2017
float3 vector_edge_integral(float3 v1, float3 v2) {
    float x = dot(v1, v2);
    float y = abs(x);

    float a = mad(mad(0.0145206, y, 0.4965155), y, 0.8543985);
    float b = mad((4.1616724 + y), y, 3.4175940);
    float v = a / b;

    float theta_over_sintheta = (x > 0.0) ? v : 0.5 * rsqrt(max(1.0 - x * x, 1e-7)) - v;

    return cross(v1, v2) * theta_over_sintheta;
}

float edge_integral(float3 v1, float3 v2) { return vector_edge_integral(v1, v2).z; }

float3 evaluate_triangle_light(IsotropicLTC bsdf, float3 wo, float3 position, float3 normal, float3 positions[3], float3 emission[3], bool two_sided) {
    // Rotate area light in (T1, T2, N) basis
    float3x3 ltc_basis = ltc_shading_space(wo, normal);

    float3x3 M_inverse = mul(bsdf.get_inverse_M(), ltc_basis);

    float3 ltc_vertices[4];
    ltc_vertices[0] = mul(M_inverse, positions[0] - position);
    ltc_vertices[1] = mul(M_inverse, positions[1] - position);
    ltc_vertices[2] = mul(M_inverse, positions[2] - position);
    ltc_vertices[3] = ltc_vertices[0];

    int vertex_count = clip_triangle_to_horizon(ltc_vertices);
    if (vertex_count == 0)
        // Early out if the entire light is clipped.
        return float3(0, 0, 0);

    // Project vertices onto sphere
    ltc_vertices[0] = normalize(ltc_vertices[0]);
    ltc_vertices[1] = normalize(ltc_vertices[1]);
    ltc_vertices[2] = normalize(ltc_vertices[2]);
    ltc_vertices[3] = vertex_count == 4 ? normalize(ltc_vertices[3]) : ltc_vertices[0];

    // Integrate triangle over cosine distribution.
    float3 F = float3(0, 0, 0);
    F += vector_edge_integral(ltc_vertices[0], ltc_vertices[1]);
    F += vector_edge_integral(ltc_vertices[1], ltc_vertices[2]);
    F += vector_edge_integral(ltc_vertices[2], ltc_vertices[3]);
    if (vertex_count == 4)
        F += vector_edge_integral(ltc_vertices[3], ltc_vertices[0]);
    float integral = two_sided ? abs(F.z) : max(0.0, -F.z); // Negate integral due to winding order.

    float3 point_emission = emission[0];
    if (any(emission[0] != emission[1]) || any(emission[0] != emission[2])) {
        // F points to the polygon and should always intersect.
        // Slide 105 in Real-Time Area Lighting: a Journey from Research to Production
        // TODO Handle triangle clipping, as the fourth vertex isn't accounted for and color indices and vertex indices don't line up.
        float3 triangle_vertices[] = { ltc_vertices[0], ltc_vertices[1], ltc_vertices[2] };
        float3 barycentric_coord;
        if (!ray_triangle_intersection(float3(0, 0, 0), -F, triangle_vertices, barycentric_coord))
            barycentric_coord = project_barycentric_coords_to_triangle_coarse(barycentric_coord);
        point_emission = emission[0] * barycentric_coord.x + emission[1] * barycentric_coord.y + emission[2] * barycentric_coord.z;
    }

    return integral * point_emission;
}

float evaluate_triangle_light(IsotropicLTC bsdf, float3 wo, float3 position, float3 normal, float3 positions[3], bool two_sided) {
    float3 emission[3] = { float3(1,1,1), float3(1,1,1), float3(1,1,1) };
    return evaluate_triangle_light(bsdf, wo, position, normal, positions, emission, two_sided).r;
}

// Evalaute a triangle light on a lambertian surface.
float evaluate_triangle_light_lambert(float3 wo, float3 position, float3 normal, float3 positions[3], bool two_sided) {
    return evaluate_triangle_light(IsotropicLTC::identity(), wo, position, normal, positions, two_sided).r;
}

}

#endif // _DX11_RENDERER_SHADERS_LTC_AREA_LIGHT_H_