// Linear Transformed Cosine light source application.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include "Utils.hlsl"

#ifndef _DX11_RENDERER_SHADERS_LTC_LIGHTSOURCES_H_
#define _DX11_RENDERER_SHADERS_LTC_LIGHTSOURCES_H_

namespace Ltc {

// Edge integral using the fitted function to replace acos and gain increased precision.
// Real-Time Area Lighting: a Journey from Research to Production, Stephen Hill and Eric Heitz, Siggraph, 2017
float3 vector_edge_integral(float3 v1, float3 v2) {
    float x = dot(v1, v2);
    float y = abs(x);

    float a = 0.8543985f + (0.4965155f + 0.0145206f * y) * y;
    float b = 3.4175940f + (4.1616724f + y) * y;
    float v = a / b;

    float theta_over_sintheta = (x > 0.0f) ? v : 0.5f * rsqrt(max(1.0f - x * x, 1e-7f)) - v;

    return cross(v1, v2) * theta_over_sintheta;
}

float edge_integral(float3 v1, float3 v2) { return vector_edge_integral(v1, v2).z; }

float3 evaluate_mesh_light_lambert(float3 normal, float3 wo, float3 position, float3 positions[3], float3 emission[3], bool two_sided) {
    // Construct orthonormal basis around normal with tangent pointing along the view direction.
    float3 tangent = normalize(wo - normal * dot(wo, normal));
    float3 bitangent = cross(normal, tangent);

    // Rotate area light in (T1, T2, N) basis
    float3x3 M_inverse = float3x3(tangent, bitangent, normal);
    // float3x3 M_inverse = transpose(float3x3(tangent, bitangent, normal));

    positions[0] = mul(M_inverse, positions[0] - position);
    positions[1] = mul(M_inverse, positions[1] - position);
    positions[2] = mul(M_inverse, positions[2] - position);

    // TODO triangle clipping

    // Project vertices onto sphere
    positions[0] = normalize(positions[0]);
    positions[1] = normalize(positions[1]);
    positions[2] = normalize(positions[2]);

    // Integrate triangle over cosine distribution.
    float3 F = float3(0, 0, 0);
    F += vector_edge_integral(positions[0], positions[1]);
    F += vector_edge_integral(positions[1], positions[2]);
    F += vector_edge_integral(positions[2], positions[0]);
    float integral = two_sided ? abs(F.z) : max(0.0, -F.z); // Negate integral due to winding order. TODO Fix later in MeshLightManager? Would be nice to keep this code in sync with other reference implementations

    float3 point_emission = emission[0];
    if (any(emission[0] != emission[1]) || any(emission[0] != emission[2])) {
        // F points to the polygon and should always intersect.
        // Slide 105 in Real-Time Area Lighting: a Journey from Research to Production
        float3 barycentric_coord;
        if (!ray_triangle_intersection(position, F, positions, barycentric_coord))
            barycentric_coord = project_barycentric_coords_to_triangle_coarse(barycentric_coord);
        point_emission = emission[0] * barycentric_coord.x + emission[1] * barycentric_coord.y + emission[2] * barycentric_coord.z;
    }

    return integral * point_emission;
}

float3 evaluate_mesh_light_mirror(float3 normal, float3 wo, float3 position, float3 positions[3], float3 emission[3], bool two_sided) {

    float3 wi = reflect(-wo, normal);

    float3 barycentric_coord;
    if (ray_triangle_intersection(position, wi, positions, barycentric_coord, two_sided))
        return emission[0] * barycentric_coord.x + emission[1] * barycentric_coord.y + emission[2] * barycentric_coord.z;
    else
        return float3(0, 0, 0);
}

}

#endif // _DX11_RENDERER_SHADERS_LTC_LIGHTSOURCES_H_