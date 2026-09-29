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
#include <Bifrost/Math/Disk.h>
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

_inline_all_archs_ float integrate_sphere_cap(float omega, float sigma) {
    constexpr float pi = Math::PI<float>();
    constexpr float half_pi = 0.5f * pi;

    auto safe_asinf = [](float v) -> float { return asinf(Math::clamp(v, -1.0f, 1.0f)); };

    float g = safe_asinf(cosf(sigma) / sinf(omega));
    float sin_sigma_sqrd = Math::pow2(sinf(sigma));

    auto G = [=](float omega, float sigma, float g) -> float {
        float cos_g = cos(g);
        return -2.0f * sinf(omega) * cosf(sigma) * cos_g + half_pi - g + sinf(g) * cos_g;
    };

    auto H = [=](float omega, float sigma, float g) -> float {
        float sin_sigma = sin(sigma);
        float sin_sigma_sqrd = Math::pow2(sin_sigma);
        float cos_g = cos(g);

        return cosf(omega) * (cos_g * Math::safe_sqrt(sin_sigma_sqrd - Math::pow2(cos_g)) + sin_sigma_sqrd * safe_asinf(cos_g / sin_sigma));
    };

    if (omega >= 0.0f && omega <= (half_pi - sigma))
        return pi * cosf(omega) * sin_sigma_sqrd;

    if (omega >= (half_pi - sigma) && omega < half_pi)
        return pi * cosf(omega) * sin_sigma_sqrd + G(omega, sigma, g) - H(omega, sigma, g);

    if (omega >= half_pi && omega < (half_pi + sigma))
        return G(omega, sigma, g) + H(omega, sigma, g);

    return 0.0f;
}

// Closed form solution to integrating a sphere cap against a diffuse surface.
// https://blog.selfshadow.com/publications/s2017-shading-course/heitz/s2017_pbs_ltc_lines_disks.pdf, slide 72.
_inline_all_archs_ float integrate_diffuse_sphere_cap(float average_direction_z, float form_factor) {
    float sigma = asinf(sqrtf(form_factor));
    float omega = acosf(average_direction_z);

    if (sigma <= 0.0f)
        return fmaxf(average_direction_z, 0.0f);

    float integral = integrate_sphere_cap(omega, sigma) / (Math::PI<float>() * form_factor);

    // The integral can become unstable at grazing angles or if form_factor is zero.
    // We handle that by hard setting it to zero.
    return isnan(integral) ? 0.0f : integral;
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

struct DiskControlPoints {
    Math::Vector3f center, tangent, bitangent;

    DiskControlPoints(Math::Disk disk) {
        center = disk.center;
        Math::compute_tangents(disk.normal, tangent, bitangent);
        tangent *= -disk.radius;
        bitangent *= disk.radius;
    }
};

// Find the roots of the cubic function c0 * x^3 + c1 * x^2 + c2 * x + c3.
// An extended version of the implementation from "How to solve a cubic equation, revisited"
// http://momentsingraphics.de/?p=105
// What has been extended and why is unfortunately not documented in the disk light reference code.
_inline_all_archs_ Math::Vector3f solve_cubic(float c0, float c1, float c2, float c3) {
    using namespace Bifrost::Math;

    // Normalize the polynomial by c3 and divide middle coefficients by three.
    float a = c3;
    float b = c2 / (c3 * 3.0f);
    float c = c1 / (c3 * 3.0f);
    float d = c0 / c3;

    // Compute the Hessian and the discriminant
    Vector3f delta = Vector3f(-b * b + c, -c * b + d, b * d - c * c);

    float discriminant = 4 * delta.x * delta.z - delta.y * delta.y;
    // Due to floating point precision the discriminant, while guaranteed always positive,
    // can become slightly negative. We therefore need to clamp the value to 0 before taking the squareroot.
    float sqrt_discriminant = sqrt(max(0.0f, discriminant));

    // Algorithm A
    Vector2f xlc;
    {
        float c_a = delta.x;
        float d_a = -2.0 * b * delta.x + delta.y;

        // Take the cubic root of a normalized complex number
        float theta = atan2(sqrt_discriminant, -d_a) / 3.0f;

        float x_1a = 2.0 * sqrt(-c_a) * cos(theta);
        float x_3a = 2.0 * sqrt(-c_a) * cos(theta + (2.0 / 3.0) * PIf);
        float xl = ((x_1a + x_3a) > 2.0 * b) ? x_1a : x_3a;

        xlc = Vector2f(xl - b, a);
    }

    // Algorithm D
    Vector2f xsc;
    {
        float c_d = delta.z;
        float d_d = -d * delta.y + 2.0 * c * delta.z;

        // Take the cubic root of a normalized complex number
        float theta = atan2(d * sqrt_discriminant, -d_d) / 3.0;

        float x_1d = 2.0 * sqrt(-c_d) * cos(theta);
        float x_3d = 2.0 * sqrt(-c_d) * cos(theta + (2.0 / 3.0) * PIf);
        float xs = (x_1d + x_3d < 2.0 * c) ? x_1d : x_3d;

        xsc = Vector2f(-d, xs + c);
    }

    float e = xlc.y * xsc.y;
    float f = -xlc.x * xsc.y - xlc.y * xsc.x;
    float g = xlc.x * xsc.x;
    Vector2f xmc = Vector2f(c * f - b * g, -b * f + c * e);

    return Vector3f(xsc.x / xsc.y, xmc.x / xmc.y, xlc.x / xlc.y);
}

// Evaluate a disk light wrt an LTC BRDF representation
_inline_all_archs_ float evaluate_disk_light(Math::IsotropicLTC bsdf, Math::Vector3f wo, Math::Vector3f position, Math::Vector3f normal, DiskControlPoints disk, bool is_two_sided) {
    using namespace Bifrost::Math;

    // Rotate area light into LTC basis - Slide 49
    Matrix3x3f ltc_basis = ltc_shading_space(wo, normal);
    Matrix3x3f M_inverse = bsdf.get_inverse_M() * ltc_basis;

    Vector3f center = M_inverse * (disk.center - position);
    Vector3f v1 = M_inverse * disk.tangent;
    Vector3f v2 = M_inverse * disk.bitangent;

    // Reject point behind the plane of the disk light
    if (!is_two_sided && dot(cross(v1, v2), center) < 0.0)
        return 0.0f;

    // Compute eigenvectors of ellipse - Slide 50
    float d11 = dot(v1, v1);
    float d22 = dot(v2, v2);
    float d12 = dot(v1, v2);
    // Increased the threshold to 0.0007 compared to the reference, which had it at 0.0001.
    // Increasing it improves numerical robustness and performance, as the else branch is faster.
    // Square the terms in the test to avoid a square root compared to the reference.
    float a_over_b, b;
    if (pow2(d12) / (d11 * d22) > 0.0007) { // Slide 60

        // Compute trace and determinant of 2x2 matrix
        float trace = d11 + d22;
        float determinant = -d12 * d12 + d11 * d22;

        // use sqrt matrix to solve for eigenvalues - Slide 55 - 58
        float two_sqrt_det = 2.0f * sqrt(determinant);
        float u = 0.5 * sqrt(trace - two_sqrt_det);
        float v = 0.5 * sqrt(trace + two_sqrt_det);
        float e_max = pow2(u + v);
        float e_min = pow2(u - v);

        Vector3f temp_v1, temp_v2;
        if (d11 > d22) {
            temp_v1 = v1 * d12 + v2 * (e_max - d11);
            temp_v2 = v1 * d12 + v2 * (e_min - d11);
        } else {
            temp_v1 = v2 * d12 + v1 * (e_max - d22);
            temp_v2 = v2 * d12 + v1 * (e_min - d22);
        }

        b = 1.0 / e_min;
        a_over_b = e_min / e_max; // a = 1 / e_max
        v1 = normalize(temp_v1);
        v2 = normalize(temp_v2);
    } else {
        float a = 1.0 / d11;
        b = 1.0 / d22;
        a_over_b = a * d22;
        v1 *= sqrt(a);
        v2 *= sqrt(b);
    }

    Vector3f v3 = cross(v1, v2);
    float L = dot(v3, center);
    if (L < 0.0) {
        L = -L;
        v3 = -v3;
    }

    float x0 = dot(v1, center) / L;
    float y0 = dot(v2, center) / L;

    b *= L * L;

    // We've observed that for tiny point lights, the solver returns NaN's.
    // This happens because the coefficients are so large that computing the discriminant becomes INF - INF = NAN.
    // The solution is to divide by a large factor that grows at the same rate as the polynomial coefficients.
    // We've chosen 'b', as dividing x by b cancels out with some of the terms in the coefficients and average direction computations.
    // This means that instead of solving the polynomial as defined in the reference,
    // we solve: c3/b^3 * (bx)^3 + c2/b^2 * (bx) ^2 + c1/b * bx + c0 = 0
    float recip_b = 1.0f / b;
    float c0 = a_over_b * recip_b;
    float c1 = a_over_b * (1.0 + x0 * x0 + y0 * y0) - c0 - recip_b;
    float c2 = recip_b - a_over_b * (1.0 + x0 * x0) - (1.0 + y0 * y0);
    float c3 = 1.0;

    Vector3f roots = solve_cubic(c0, c1, c2, c3); // Slide 67

    // Swizzle the roots so the negative root is in .y and gets assigned to e2
    if (roots.x < roots.y && roots.x < roots.z)
        roots = { roots.y, roots.x, roots.z };
    else if (roots.z < roots.x && roots.z < roots.y)
        roots = { roots.x, roots.z, roots.y };

    float e1 = roots.x;
    float e2 = roots.y;
    float e3 = roots.z;

    //Reference: average_direction = { a * x0 / (a - e2), b * y0 / (b - e2), 1.0 };
    Vector3f average_direction = { a_over_b * x0 / (a_over_b - e2), y0 / (1 - e2), 1.0 };
    average_direction = normalize(v1 * average_direction.x + v2 * average_direction.y + v3 * average_direction.z);

    // Calculate projected solid angle (form_factor) - Slide 72
    float l1_sqrd = -e2 / e3;
    float l2_sqrd = -e2 / e1;
    float form_factor_sqrd = (l1_sqrd * l2_sqrd) / ((1 + l1_sqrd) * (1 + l2_sqrd));
    float form_factor = sqrt(form_factor_sqrd);

    // Use horizon-clipped sphere approximation to clip the light.
    float clipping = integrate_diffuse_sphere_cap(average_direction.z, form_factor);

    return form_factor * clipping;
}

}

#endif // _BIFROST_ASSETS_SHADING_LIGHTSOURCES_LTC_AREA_LIGHT_H_