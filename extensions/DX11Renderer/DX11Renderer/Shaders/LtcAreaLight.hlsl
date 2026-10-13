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

namespace integrate_diffuse_sphere_cap_utils {

float G(float omega, float sigma, float g) {
    float cos_g = cos(g);
    return -2.0f * sin(omega) * cos(sigma) * cos_g + PI * 0.5 - g + sin(g) * cos_g;
}

float H(float omega, float sigma, float g) {
    float sin_sigma = sin(sigma);
    float sin_sigma_sqrd = pow2(sin_sigma);
    float cos_g = cos(g);

    return cos(omega) * (cos_g * sqrt(sin_sigma_sqrd - pow2(cos_g)) + sin_sigma_sqrd * asin(cos_g / sin_sigma));
}

float integrate_sphere_cap(float omega, float sigma) {
    float g = asin(cos(sigma) / sin(omega));
    float sin_sigma_sqrd = pow2(sin(sigma));

    if (omega >= 0.0f && omega <= (PI * 0.5 - sigma))
        return PI * cos(omega) * sin_sigma_sqrd;

    if (omega >= (PI * 0.5 - sigma) && omega < PI * 0.5)
        return PI * cos(omega) * sin_sigma_sqrd + G(omega, sigma, g) - H(omega, sigma, g);

    if (omega >= PI * 0.5 && omega < (PI * 0.5 + sigma))
        return G(omega, sigma, g) + H(omega, sigma, g);

    return 0.0f;
}

} // integrate_diffuse_sphere_cap_utils

// Closed form solution to integrating a sphere cap against a diffuse surface.
// https://blog.selfshadow.com/publications/s2017-shading-course/heitz/s2017_pbs_ltc_lines_disks.pdf, slide 72.
float integrate_diffuse_sphere_cap(float average_direction_z, float form_factor) {
    float sigma = asin(sqrt(form_factor));
    float omega = acos(average_direction_z);

    if (sigma <= 0.0f)
        return max(average_direction_z, 0.0f);

    float integral = integrate_diffuse_sphere_cap_utils::integrate_sphere_cap(omega, sigma) / (PI * form_factor);

    // The integral can become unstable at grazing angles or if form_factor is zero.
    // We handle that by hard setting it to zero.
    return is_NaN(integral) ? 0.0f : integral;
}

// ------------------------------------------------------------------------------------------------
// Triangle light integration
// ------------------------------------------------------------------------------------------------

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

float3 evaluate_triangle_light(IsotropicLTC bsdf, float3 wo, float3 position, float3 normal, float3 positions[3], float3 emitted_radiance[3], bool two_sided) {
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

    float3 point_emitted_radiance = emitted_radiance[0];
    if (any(emitted_radiance[0] != emitted_radiance[1]) || any(emitted_radiance[0] != emitted_radiance[2])) {
        // F points to the polygon and should always intersect.
        // Slide 105 in Real-Time Area Lighting: a Journey from Research to Production
        // TODO Handle triangle clipping, as the fourth vertex isn't accounted for and color indices and vertex indices don't line up.
        float3 triangle_vertices[] = { ltc_vertices[0], ltc_vertices[1], ltc_vertices[2] };
        float3 barycentric_coord;
        if (!ray_triangle_intersection(float3(0, 0, 0), -F, triangle_vertices, barycentric_coord))
            barycentric_coord = project_barycentric_coords_to_triangle_coarse(barycentric_coord);
        point_emitted_radiance = emitted_radiance[0] * barycentric_coord.x + emitted_radiance[1] * barycentric_coord.y + emitted_radiance[2] * barycentric_coord.z;
    }

    return integral * point_emitted_radiance;
}

float evaluate_triangle_light(IsotropicLTC bsdf, float3 wo, float3 position, float3 normal, float3 positions[3], bool two_sided) {
    float3 emitted_radiance[3] = { float3(1,1,1), float3(1,1,1), float3(1,1,1) };
    return evaluate_triangle_light(bsdf, wo, position, normal, positions, emitted_radiance, two_sided).r;
}

// Evalaute a triangle light on a lambertian surface.
float evaluate_triangle_light_lambert(float3 wo, float3 position, float3 normal, float3 positions[3], bool two_sided) {
    return evaluate_triangle_light(IsotropicLTC::identity(), wo, position, normal, positions, two_sided).r;
}

// ------------------------------------------------------------------------------------------------
// Disk light integration based on
// Real-Time Line- and Disk-Light Shading with Linearly Transformed Cosines, Heitz and Hill, 2017.
// https://blog.selfshadow.com/publications/s2017-shading-course/heitz/s2017_pbs_ltc_lines_disks.pdf
// Disk lights start on slide 38.
// Reference code by Stephen Hill can be found in
// https://github.com/selfshadow/ltc_code/blob/master/webgl/shaders/ltc/ltc_disk.fs
// ------------------------------------------------------------------------------------------------

// Find the roots of the cubic function c3 * x^3 + c2 * x^2 + c1 * x + c0.
// An extended version of the implementation from "How to solve a cubic equation, revisited"
// http://momentsingraphics.de/?p=105
// What has been extended and why is unfortunately not documented in the disk light reference code.
float3 solve_cubic(float c0, float c1, float c2, float c3) {
    // Normalize the polynomial by c3 and divide middle coefficients by three.
    float a = c3;
    float b = c2 / (c3 * 3.0);
    float c = c1 / (c3 * 3.0);
    float d = c0 / c3;

    // Compute the Hessian and the discriminant
    float3 delta = float3(-b * b + c, -c * b + d, b * d - c * c);

    float discriminant = 4 * delta.x * delta.z - delta.y * delta.y;
    // Due to floating point precision the discriminant, while guaranteed always positive,
    // can become slightly negative. We therefore need to clamp the value to 0 before taking the square root.
    float sqrt_discriminant = safe_sqrt(discriminant);

    // Algorithm A
    float2 xlc;
    {
        float c_a = delta.x;
        float d_a = -2.0 * b * delta.x + delta.y;

        // Take the cubic root of a normalized complex number
        float theta = atan2(sqrt_discriminant, -d_a) / 3.0f;

        float x_1a = 2.0 * sqrt(-c_a) * cos(theta);
        float x_3a = 2.0 * sqrt(-c_a) * cos(theta + (2.0 / 3.0) * PI);

        float xl = ((x_1a + x_3a) > 2.0 * b) ? x_1a : x_3a;

        xlc = float2(xl - b, a);
    }

    // Algorithm D
    float2 xsc;
    {
        float c_d = delta.z;
        float d_d = -d * delta.y + 2.0 * c * delta.z;

        // Take the cubic root of a normalized complex number
        float theta = atan2(d * sqrt_discriminant, -d_d) / 3.0;

        float x_1d = 2.0 * sqrt(-c_d) * cos(theta);
        float x_3d = 2.0 * sqrt(-c_d) * cos(theta + (2.0 / 3.0) * PI);

        float xs = (x_1d + x_3d < 2.0 * c) ? x_1d : x_3d;

        xsc = float2(-d, xs + c);
    }

    float e = xlc.y * xsc.y;
    float f = -xlc.x * xsc.y - xlc.y * xsc.x;
    float g = xlc.x * xsc.x;
    float2 xmc = float2(c * f - b * g, -b * f + c * e);

    float3 roots = float3(xsc.x / xsc.y, xmc.x / xmc.y, xlc.x / xlc.y);
    if (roots.x < roots.y && roots.x < roots.z)
        return float3(roots.y, roots.x, roots.z);
    else if (roots.z < roots.x && roots.z < roots.y)
        return float3(roots.x, roots.z, roots.y);
    else
        return roots;
}

// Evaluate a disk light wrt an LTC BRDF representation
// The disk light control points are [center, radius * -tangent, radius * bitangent]
float evaluate_disk_light(IsotropicLTC bsdf, float3 wo, float3 position, float3 normal, float3 disk_control_points[3], bool is_two_sided) {
    // Rotate area light into LTC basis - Slide 49
    float3x3 ltc_basis = ltc_shading_space(wo, normal);
    float3x3 M_inverse = mul(bsdf.get_inverse_M(), ltc_basis);

    float3 center = mul(M_inverse, disk_control_points[0] - position);
    float3 v1 = mul(M_inverse, disk_control_points[1]);
    float3 v2 = mul(M_inverse, disk_control_points[2]);

    // Reject point in the plane of the disk light or behind it if the light is one-sided.
    float light_orientation = dot(cross(v1, v2), center);
    if (light_orientation == 0 || (!is_two_sided && light_orientation < 0.0))
        return 0.0f;

    // Compute eigenvectors of ellipse - Slide 50
    float d11 = dot(v1, v1);
    float d22 = dot(v2, v2);
    float d12 = dot(v1, v2);
    // Increased the threshold to 0.0007 compared to the reference, which had it at 0.0001.
    // Increasing it improves numerical robustness and performance, as the else branch is faster.
    // Square the terms in the test to avoid a square root compared to the reference.
    float a_over_b, b;
    if (pow2(d12) / (d11 * d22) > 5e-7) { // Slide 60

        // Compute trace and determinant of 2x2 matrix
        float trace = d11 + d22;
        float determinant = -d12 * d12 + d11 * d22;

        // use sqrt matrix to solve for eigenvalues - Slide 55 - 58
        float two_sqrt_det = 2.0f * sqrt(determinant);
        float u = 0.5 * sqrt(trace - two_sqrt_det);
        float v = 0.5 * sqrt(trace + two_sqrt_det);
        float e_max = pow2(u + v);
        float e_min = pow2(u - v);

        float3 temp_v1, temp_v2;
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

    float3 v3 = cross(v1, v2);
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

    float3 roots = solve_cubic(c0, c1, c2, c3); // Slide 67
    float e1 = roots.x;
    float e2 = roots.y;
    float e3 = roots.z;

    //Reference: average_direction = { a * x0 / (a - e2), b * y0 / (b - e2), 1.0 };
    float3 average_direction = { a_over_b * x0 / (a_over_b - e2), y0 / (1 - e2), 1.0 };
    average_direction = normalize(v1 * average_direction.x + v2 * average_direction.y + v3 * average_direction.z);

    // Calculate projected solid angle (form_factor) - Slide 72
    float l1_sqrd = -e2 / e3;
    float l2_sqrd = -e2 / e1;
    float form_factor_sqrd = (l1_sqrd * l2_sqrd) / ((1 + l1_sqrd) * (1 + l2_sqrd));
    float form_factor = safe_sqrt(form_factor_sqrd);

    // Use horizon-clipped sphere approximation to clip the light.
    float clipping = integrate_diffuse_sphere_cap(average_direction.z, form_factor);

    return form_factor * clipping;
}

}

#endif // _DX11_RENDERER_SHADERS_LTC_AREA_LIGHT_H_