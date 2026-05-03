// Bifrost area light approximations using linear transformed cosines.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_LIGHTSOURCES_LTC_AREA_LIGHT_H_
#define _BIFROST_ASSETS_SHADING_LIGHTSOURCES_LTC_AREA_LIGHT_H_

#include <Bifrost/Math/LTC.h>

namespace Bifrost::Assets::Shading::LightSources::LtcAreaLight {

_inline_all_archs_ Math::Matrix3x3f ltc_shading_space(Math::Vector3f wo, Math::Vector3f normal) {
    float wo_dot_n = dot(wo, normal);
    Math::Vector3f tangent, bitangent;
    if (abs(wo_dot_n) <= 0.999999f) {
        // Construct orthonormal basis around normal with tangent pointing along the view direction.
        tangent = Math::normalize(wo - normal * wo_dot_n);
        bitangent = Math::cross(normal, tangent);
    } else
        // Edge case where wo is so similar to the normal that the forward direction is unstable
        // and we just compute some tangent space around the normal.
        compute_tangents(normal, tangent, bitangent);

    return Math::Matrix3x3f({ tangent, bitangent, normal });
}

_inline_all_archs_ Math::Vector3f intersection_at_horizon(Math::Vector3f v0, Math::Vector3f v1) {
    float t = -v0.z / (v1.z - v0.z);
    return { Math::lerp(v0.x, v1.x, t), Math::lerp(v0.y, v1.y, t), 0.0f };
}

// Clips the triangle against the horizon. Returns the number of valid vertices returned in the input array.
// If one vertex is below the horizon, then clipping returns a four vertex polygon and the fourth vertex is populated.
// If all vertices are below the horizon then the 0 is returned.
_inline_all_archs_ int clip_triangle_to_horizon(Math::Vector3f vertices[4]) {
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

    Math::Vector3f v0_to_v1 = intersection_at_horizon(vertices[0], vertices[1]);
    Math::Vector3f v0_to_v2 = intersection_at_horizon(vertices[0], vertices[2]);
    Math::Vector3f v1_to_v2 = intersection_at_horizon(vertices[1], vertices[2]);

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
_inline_all_archs_ Math::Vector3f vector_edge_integral(Math::Vector3f v1, Math::Vector3f v2) {
    float x = dot(v1, v2);
    float y = abs(x);

    float a = 0.8543985f + (0.4965155f + 0.0145206f * y) * y;
    float b = 3.4175940f + (4.1616724f + y) * y;
    float v = a / b;

    float theta_over_sintheta = (x > 0.0f) ? v : 0.5f / sqrt(fmaxf(1.0f - x * x, 1e-7f)) - v;

    return cross(v1, v2) * theta_over_sintheta;
}

_inline_all_archs_ float edge_integral(Math::Vector3f v1, Math::Vector3f v2) { return vector_edge_integral(v1, v2).z; }

_inline_all_archs_ float evaluate_triangle_light(Math::IsotropicLTC bsdf, Math::Vector3f wo, Math::Vector3f position, Math::Vector3f normal, const Math::Vector3f light_vertices[3], bool two_sided) {
    using namespace Bifrost::Math;

    Matrix3x3f ltc_basis = ltc_shading_space(wo, normal);

    // Rotate area light into shading basis
    Matrix3x3f M_inverse = bsdf.get_inverse_M() * ltc_basis;

    Vector3f ltc_vertices[4];
    ltc_vertices[0] = M_inverse * (light_vertices[0] - position);
    ltc_vertices[1] = M_inverse * (light_vertices[1] - position);
    ltc_vertices[2] = M_inverse * (light_vertices[2] - position);

    int vertex_count = clip_triangle_to_horizon(ltc_vertices);
    if (vertex_count == 0)
        // Early out if the entire light is clipped.
        return 0.0f;

    // Project vertices onto sphere
    ltc_vertices[0] = normalize(ltc_vertices[0]);
    ltc_vertices[1] = normalize(ltc_vertices[1]);
    ltc_vertices[2] = normalize(ltc_vertices[2]);
    ltc_vertices[3] = vertex_count == 4 ? normalize(ltc_vertices[3]) : ltc_vertices[0];

    // Integrate triangle over cosine distribution.
    Vector3f F = { 0, 0, 0 };
    F += vector_edge_integral(ltc_vertices[0], ltc_vertices[1]);
    F += vector_edge_integral(ltc_vertices[1], ltc_vertices[2]);
    F += vector_edge_integral(ltc_vertices[2], ltc_vertices[3]);
    if (vertex_count == 4)
        F += vector_edge_integral(ltc_vertices[3], ltc_vertices[0]);
    float integral = two_sided ? abs(F.z) : fmaxf(0.0, -F.z); // Negate integral due to winding order.

    return integral;
}

_inline_all_archs_ Math::RGB evaluate_triangle_light(Math::IsotropicLTC bsdf, Math::Vector3f wo, Math::Vector3f position, Math::Vector3f normal, const Math::Vector3f light_vertices[3], Math::RGB emission, bool two_sided) {
    return emission * evaluate_triangle_light(bsdf, wo, position, normal, light_vertices, two_sided);
}

_inline_all_archs_ float evaluate_triangle_light_lambert(Math::Vector3f wo, Math::Vector3f position, Math::Vector3f normal, Math::Vector3f light_vertices[3], bool two_sided) {
    return evaluate_triangle_light(Math::IsotropicLTC::identity(), wo, position, normal, light_vertices, two_sided);
}

}

#endif // _BIFROST_ASSETS_SHADING_LIGHTSOURCES_LTC_AREA_LIGHT_H_