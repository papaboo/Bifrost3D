// Bifrost area light approximations using linear transformed cosines.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_LIGHTSOURCES_LTC_AREA_LIGHT_H_
#define _BIFROST_ASSETS_SHADING_LIGHTSOURCES_LTC_AREA_LIGHT_H_

#include <Bifrost/Assets/Shading/Constants.h>
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

// Approximation to the sphere clipping integral.
// Less smooth than the texture and have energy issues.
_inline_all_archs_ float approximate_diffuse_sphere_integral(float average_direction_z, float form_factor) {
    // Cheap approximation. Less smooth and have energy issues.
    return fmaxf((form_factor * form_factor + average_direction_z) / (form_factor + 1.0f), 0.0f);
}

// ------------------------------------------------------------------------------------------------
// Triangle light integration
// ------------------------------------------------------------------------------------------------

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

_inline_all_archs_ float evaluate_triangle_light(Math::IsotropicLTC bsdf, Math::Vector3f wo, Math::Vector3f position, Math::Vector3f normal, const Math::Vector3f light_vertices[3], bool two_sided) {
    using namespace Bifrost::Math;

    Matrix3x3f ltc_basis = ltc_shading_space(wo, normal);
    Matrix3x3f M_inverse = bsdf.get_inverse_M() * ltc_basis;

    // Transform triangle light into shading basis
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

_inline_all_archs_ Math::RGB evaluate_triangle_light(Math::IsotropicLTC bsdf, Math::Vector3f wo, Math::Vector3f position, Math::Vector3f normal, const Math::Vector3f light_vertices[3], Math::RGB emitted_radiance, bool two_sided) {
    return emitted_radiance * evaluate_triangle_light(bsdf, wo, position, normal, light_vertices, two_sided);
}

_inline_all_archs_ float evaluate_triangle_light_lambert(Math::Vector3f wo, Math::Vector3f position, Math::Vector3f normal, Math::Vector3f light_vertices[3], bool two_sided) {
    return evaluate_triangle_light(Math::IsotropicLTC::identity(), wo, position, normal, light_vertices, two_sided);
}

// ------------------------------------------------------------------------------------------------
// Disk light integration based on
// Real-Time Line- and Disk-Light Shading with Linearly Transformed Cosines, Heitz and Hill, 2017.
// https://blog.selfshadow.com/publications/s2017-shading-course/heitz/s2017_pbs_ltc_lines_disks.pdf
// Disk lights start on slide 38.
// Reference code by Stephen Hill can be found in
// https://github.com/selfshadow/ltc_code/blob/master/webgl/shaders/ltc/ltc_disk.fs
// ------------------------------------------------------------------------------------------------

// Find the roots of the cubic function c0 + c1 * x + c2 * x^2 + c3 * x^3.
// An extended version of the implementation from "How to solve a cubic equation, revisited"
// http://momentsingraphics.de/?p=105
// What has been extended and why is unfortunately not documented in the disk light reference code.
_inline_all_archs_ Math::Vector3f solve_cubic(Math::Vector4f Coefficient) {
    using namespace Bifrost::Math;

    // Normalize the polynomial
    Coefficient.x /= Coefficient.w;
    Coefficient.y /= Coefficient.w;
    Coefficient.z /= Coefficient.w;
    // Divide middle coefficients by three
    Coefficient.y /= 3.0; // TODO inline with division above
    Coefficient.z /= 3.0;

    float A = Coefficient.w;
    float B = Coefficient.z;
    float C = Coefficient.y;
    float D = Coefficient.x;

    // Compute the Hessian and the discriminant
    Vector3f Delta = Vector3f(
        -Coefficient.z * Coefficient.z + Coefficient.y,
        -Coefficient.y * Coefficient.z + Coefficient.x,
        dot(Vector2f(Coefficient.z, -Coefficient.y), Vector2f(Coefficient.x, Coefficient.y))
    );

    float Discriminant = dot(Vector2f(4.0 * Delta.x, -Delta.y), Vector2f(Delta.z, Delta.y));
    // Due to floating point precision the discrimiant, while guaranteed always positive,
    // can become slightly negative. This results in NANs below when taking the square root.
    Discriminant = max(0.0f, Discriminant);

    Vector3f RootsA, RootsD;

    Vector2f xlc, xsc;

    // Algorithm A
    {
        float A_a = 1.0;
        float C_a = Delta.x;
        float D_a = -2.0 * B * Delta.x + Delta.y;

        // Take the cubic root of a normalized complex number
        float Theta = atan2(sqrt(Discriminant), -D_a) / 3.0f;

        float x_1a = 2.0 * sqrt(-C_a) * cos(Theta);
        float x_3a = 2.0 * sqrt(-C_a) * cos(Theta + (2.0 / 3.0) * PIf);

        float xl;
        if ((x_1a + x_3a) > 2.0 * B)
            xl = x_1a;
        else
            xl = x_3a;

        xlc = Vector2f(xl - B, A);
    }

    // Algorithm D
    {
        float A_d = D;
        float C_d = Delta.z;
        float D_d = -D * Delta.y + 2.0 * C * Delta.z;

        // Take the cubic root of a normalized complex number
        float Theta = atan2(D * sqrt(Discriminant), -D_d) / 3.0;

        float x_1d = 2.0 * sqrt(-C_d) * cos(Theta);
        float x_3d = 2.0 * sqrt(-C_d) * cos(Theta + (2.0 / 3.0) * PIf);

        float xs;
        if (x_1d + x_3d < 2.0 * C)
            xs = x_1d;
        else
            xs = x_3d;

        xsc = Vector2f(-D, xs + C);
    }

    float E = xlc.y * xsc.y;
    float F = -xlc.x * xsc.y - xlc.y * xsc.x;
    float G = xlc.x * xsc.x;

    Vector2f xmc = Vector2f(C * F - B * G, -B * F + C * E);

    Vector3f Root = Vector3f(xsc.x / xsc.y, xmc.x / xmc.y, xlc.x / xlc.y);

    if (Root.x < Root.y && Root.x < Root.z)
        // Root.xyz = Root.yxz;
        return { Root.y, Root.x, Root.z };
    else if (Root.z < Root.x && Root.z < Root.y)
        // Root.xyz = Root.xzy;
        return { Root.x, Root.z, Root.y };
    else 
        return Root;
}

// Find the roots of the cubic function c0 + c1 * x + c2 * x^2 + c3 * x^3.
_inline_all_archs_ Math::Vector3f solve_cubic(float c0, float c1, float c2, float c3) { return solve_cubic(Math::Vector4f(c0, c1, c2, c3)); }

// Evaluate a disk light wrt an LTC BRDF representation
// The disk light control points are [center, radius * -tangent, radius * bitangent]
_inline_all_archs_ float evaluate_disk_light(Math::IsotropicLTC bsdf, Math::Vector3f wo, Math::Vector3f position, Math::Vector3f normal, Math::Vector3f disk_control_points[3], bool is_two_sided) {
    using namespace Bifrost::Math;

    // Compute control points on the ellipse.
    Vector3f center = disk_control_points[0] - position;
    Vector3f tangent_point = center + disk_control_points[1];
    Vector3f bitangent_point = center + disk_control_points[2];

    // Rotate area light into LTC basis - Slide 49
    Matrix3x3f ltc_basis = ltc_shading_space(wo, normal);
    Matrix3x3f M_inverse = bsdf.get_inverse_M() * ltc_basis;

    Vector3f C = M_inverse * center;
    Vector3f V1 = M_inverse * tangent_point - C;
    Vector3f V2 = M_inverse * bitangent_point - C;

    // Reject point behind the plane of the disk light
    if (!is_two_sided && dot(cross(V1, V2), C) < 0.0)
        return 0.0f;

    // Compute eigenvectors of ellipse - Slide 50
    float a, b;
    float d11 = dot(V1, V1);
    float d22 = dot(V2, V2);
    float d12 = dot(V1, V2);
    // Increased the threshold compared to the reference, which had it at 0.0001.
    // Increasing it improves numerical robustness and performance, as the else branch is faster.
    // TODO This is never true for any of the tests. I need tests that exercise this path.
    // TODO we can square this and avoid the sqrt
    if (abs(d12) / sqrt(d11 * d22) > 0.0007) // Slide 60
    {
        // Compute trace and determinant of 2x2 matrix
        float trace = d11 + d22;
        float determinant = -d12 * d12 + d11 * d22;

        // use sqrt matrix to solve for eigenvalues - Slide 55 - 58
        float two_sqrt_det = 2.0f * sqrt(determinant);
        float u = 0.5 * sqrt(trace - two_sqrt_det);
        float v = 0.5 * sqrt(trace + two_sqrt_det);
        float e_max = pow2(u + v);
        float e_min = pow2(u - v);

        Vector3f V1_, V2_;
        if (d11 > d22)
        {
            V1_ = V1 * d12 + V2 * (e_max - d11);
            V2_ = V1 * d12 + V2 * (e_min - d11);
        } else {
            V1_ = V2 * d12 + V1 * (e_max - d22);
            V2_ = V2 * d12 + V1 * (e_min - d22);
        }

        a = 1.0 / e_max;
        b = 1.0 / e_min;
        V1 = normalize(V1_);
        V2 = normalize(V2_);
    } else {
        a = 1.0 / dot(V1, V1); // TODO use d11 and d22 above
        b = 1.0 / dot(V2, V2);
        V1 *= sqrt(a);
        V2 *= sqrt(b);
    }

    Vector3f V3 = cross(V1, V2);
    if (dot(C, V3) < 0.0)
        V3 *= -1.0;

    float L = dot(V3, C); // TODO Reuse dot product from above
    float x0 = dot(V1, C) / L;
    float y0 = dot(V2, C) / L;

    a *= L * L;
    b *= L * L;

    float c0 = a * b;
    float c1 = a * b * (1.0 + x0 * x0 + y0 * y0) - a - b;
    float c2 = 1.0 - a * (1.0 + x0 * x0) - b * (1.0 + y0 * y0);
    float c3 = 1.0;

    Vector3f roots = solve_cubic(c0, c1, c2, c3); // Slide 67
    float e1 = roots.x;
    float e2 = roots.y;
    float e3 = roots.z;

    Vector3f average_direction = Vector3f(a * x0 / (a - e2), b * y0 / (b - e2), 1.0);

    Matrix3x3f rotate;
    rotate.set_column(0, V1);
    rotate.set_column(1, V2);
    rotate.set_column(2, V3);

    average_direction = normalize(rotate * average_direction);

    // Calculate projected solid angle (form_factor) - Slide 72
    float L1 = sqrt(-e2 / e3); // TODO Can I remove the square roots if I compute the form factor squared and then take the sqrt?
    float L2 = sqrt(-e2 / e1);
    float form_factor = L1 * L2 * 1.0f / sqrt((1.0 + L1 * L1) * (1.0 + L2 * L2));

    // Use horizon-clipped sphere approximation to clip the light.
    // float clipping_approximation = precomputed_diffuse_sphere_integral(average_direction.z, form_factor);
    float clipping_approximation = approximate_diffuse_sphere_integral(average_direction.z, form_factor);

    return form_factor * clipping_approximation;
}

}

#endif // _BIFROST_ASSETS_SHADING_LIGHTSOURCES_LTC_AREA_LIGHT_H_