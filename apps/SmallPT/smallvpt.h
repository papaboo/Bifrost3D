// Smallvpt
// Based on https://github.com/seifeddinedridi/smallvpt
// ----------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _SMALL_VPT_H_
#define _SMALL_VPT_H_

#pragma warning(disable: 4244) // Disable double to float warning

#include <Bifrost/Math/Color.h>
#include <Bifrost/Math/Utils.h>
#include <Bifrost/Math/Vector.h>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace Bifrost::Math;

namespace smallvpt {

namespace XORShift { // XOR shift PRNG
unsigned int x = 123456789;
unsigned int y = 362436069;
unsigned int z = 521288629;
unsigned int w = 88675123;
inline float frand() {
    unsigned int t;
    t = x ^ (x << 11);
    x = y; y = z; z = w;
    return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8))) * (1.0f / 4294967295.0f);
}
};

struct Ray { Vector3d origin, direction; Ray() {} Ray(Vector3d o, Vector3d d) : origin(o), direction(d) {} };
enum class BSDF { Diffuse, Specular, Glass };

struct Sphere {
    double radius;
    Vector3d position;
    RGB emission, albedo;
    BSDF bsdf;
    Sphere(double r, Vector3d p, RGB e, RGB a, BSDF bsdf) :
        radius(r), position(p), emission(e), albedo(a), bsdf(bsdf) {}
    double intersect(const Ray &r, double *tin = NULL, double *tout = NULL) const { // returns distance, 0 if nohit
        Vector3d op = position - r.origin; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        double t, eps = 1e-4, b = dot(op, r.direction), det = b * b - dot(op, op) + radius * radius;
        if (det < 0) return 0; else det = sqrt(det);
        if (tin && tout) { *tin = (b - det <= 0) ? 0 : b - det; *tout = b + det; }
        return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
    }
};

Sphere spheres[] = { // Scene: radius, position, emission, color, material
    Sphere(1e5, Vector3d(1e5 + 1, 40.8, 81.6), RGB::black(), RGB(.75f, .25f, .25f), BSDF::Diffuse), // Left
    Sphere(1e5, Vector3d(-1e5 + 99, 40.8, 81.6), RGB::black(), RGB(.25f, .25f, .75f), BSDF::Diffuse), // Right
    Sphere(1e5, Vector3d(50, 40.8, 1e5), RGB::black(), RGB(.75f), BSDF::Diffuse), // Back
    Sphere(1e5, Vector3d(50, 40.8, -1e5 + 170), RGB::black(), RGB::black(), BSDF::Diffuse), // Front
    Sphere(1e5, Vector3d(50, 1e5, 81.6), RGB::black(), RGB(.75f), BSDF::Diffuse), // Bottom
    Sphere(1e5, Vector3d(50, -1e5 + 81.6, 81.6), RGB::black(), RGB(.75f), BSDF::Diffuse), // Top
    Sphere(16.5, Vector3d(27, 16.5, 47), RGB::black(), RGB(.999f), BSDF::Specular), // Mirror
    Sphere(16.5, Vector3d(73, 16.5, 78), RGB::black(), RGB(.999f), BSDF::Glass), // Glass
    Sphere(600, Vector3d(50, 681.6 - .27, 81.6), RGB(12.0f), RGB::black(), BSDF::Diffuse) // Light
};

Sphere homogeneousMedium(300, Vector3d(50, 50, 80), RGB::black(), RGB::black(), BSDF::Diffuse);

const double sigma_s = 0.009, sigma_a = 0.006, sigma_t = sigma_s + sigma_a;
inline bool intersect_scene(const Ray &r, double &t, int &id, double tmax = 1e20) {
    int scene_object_count = sizeof(spheres) / sizeof(Sphere);
    double d;
    t = tmax;
    for (int i = 0; i < scene_object_count; ++i)
        if ((d = spheres[i].intersect(r)) && d < t) {
            t = d;
            id = i;
        }
    return t < tmax;
}
inline float sampleSegment(double epsilon, float sigma, float smax) {
    return -logf(1.0f - epsilon * (1.0f - expf(-sigma * smax))) / sigma;
}
inline Vector3f sampleHG(float g, float e1, float e2) {
    float s = 1.0f - 2.0f * e1;
    float cost = (s + 2.0f * g * g * g * (-1.0f + e1) * e1 + g * g * s + 2.0f * g * (1.0f - e1 + e1 * e1)) / ((1.0f + g * s) * (1.0f + g * s));
    float sint = sqrt(1.0f - cost * cost);
    return Vector3f(cosf(2.0f * M_PI * e2) * sint, sinf(2.0f * M_PI * e2) * sint, cost);
}

inline double scatter(const Ray &ray, Ray &scattering_ray, double tin, float tout, double &s) {
    double t_distance = tout - tin;
    s = sampleSegment(XORShift::frand(), float(sigma_s), float(t_distance));
    Vector3d scattering_position = ray.origin + ray.direction * (tin + s);
    Vector3f local_dir = sampleHG(-0.5, XORShift::frand(), XORShift::frand()); //Sample a direction ~ Henyey-Greenstein's phase function
    Vector3f tangent, bitangent;
    compute_tangents((Vector3f)ray.direction, tangent, bitangent);
    Vector3d ray_dir = Vector3d(tangent * local_dir.x + bitangent * local_dir.y) + ray.direction * local_dir.z;
    scattering_ray = Ray(scattering_position, ray_dir);
    return 1.0 - exp(-sigma_s * t_distance);
}
RGB integrate_radiance(const Ray &r, int depth) {
    // Avoid stack overflow from recursion
    if (depth > 250)
        return RGB::black();

    double ray_t;
    int scene_object_id = 0;
    double tnear, tfar, recip_surface_interaction_probability = 1.0, absorption = 1.0;
    bool inside_medium = homogeneousMedium.intersect(r, &tnear, &tfar) > 0;
    if (inside_medium) {
        Ray sRay;
        double s, ms = scatter(r, sRay, tnear, tfar, s), scattering_probability = ms;
        if (XORShift::frand() <= scattering_probability) {
            // Sample volume
            if (!intersect_scene(r, ray_t, scene_object_id, tnear + s))
                return integrate_radiance(sRay, ++depth) * (ms / scattering_probability);
        } else {
            // Sample surface
            double surface_interaction_probability = 1.0 - scattering_probability;
            recip_surface_interaction_probability = 1.0 / surface_interaction_probability;
            if (!intersect_scene(r, ray_t, scene_object_id))
                return RGB::black();
        }
        if (ray_t >= tnear) {
            double dist = (ray_t > tfar ? tfar - tnear : ray_t - tnear);
            absorption = exp(-sigma_t * dist);
        }
    } else
        if (!intersect_scene(r, ray_t, scene_object_id))
            return RGB::black();

    const Sphere& scene_object = spheres[scene_object_id];
    Vector3d x = r.origin + r.direction * ray_t;
    Vector3d n = normalize(x - scene_object.position);
    Vector3d nl = dot(n, r.direction) < 0 ? n : n * -1;
    RGB albedo = scene_object.albedo, emission = scene_object.emission;

    // Scale the ray response by the absorption
    albedo = albedo * absorption;
    emission = scene_object.emission * absorption;

    // Russian roulette after 5 interactions.
    // The decision is taken based on local albedo information.
    // If the ray passes, then albedo is scaled by the russian roulette probability and tracing continues.
    // The material's emission is always applied and therefore not scaled by russian roulette probability.
    if (++depth > 5) {
        float max_albedo_value = max(albedo.r, max(albedo.g, albedo.b));
        if (XORShift::frand() < max_albedo_value)
            albedo *= 1.0f / max_albedo_value;
        else
            return emission;
    }

    if (scene_object.bsdf == BSDF::Diffuse) {
        float r1 = 2 * float(M_PI) * XORShift::frand();
        float r2 = XORShift::frand();
        float r2s = sqrt(r2);
        Vector3f normal = Vector3f(nl);
        Vector3f tangent, bitangent;
        compute_tangents(normal, tangent, bitangent);
        Vector3d d = (Vector3d)normalize(tangent * cosf(r1) * r2s + bitangent * sinf(r1) * r2s + normal * sqrt(1 - r2));
        return (emission + albedo * integrate_radiance(Ray(x, d), depth)) * recip_surface_interaction_probability;
    }

    Ray reflection_ray(x, reflect(r.direction, n));
    if (scene_object.bsdf == BSDF::Specular)
        return (emission + albedo * integrate_radiance(reflection_ray, depth)) * recip_surface_interaction_probability;

    // Glass
    bool into = dot(n, nl) > 0; // Ray from outside going in?
    double air_ior = 1;
    double glass_ior = 1.5;
    double relative_ior = into ? air_ior / glass_ior : glass_ior / air_ior;
    double ddn = dot(r.direction, nl);
    double cos2t;
    if ((cos2t = 1 - relative_ior * relative_ior * (1 - ddn * ddn)) < 0)    // Total internal reflection
        return (emission + integrate_radiance(reflection_ray, depth)) * recip_surface_interaction_probability;
    Vector3d tdir = normalize(r.direction * relative_ior - n * ((into ? 1 : -1)*(ddn * relative_ior + sqrt(cos2t))));
    double specularity = dielectric_specularity(air_ior, glass_ior);
    double cos_theta = into ? -ddn : dot(n, tdir);
    double Re = schlick_fresnel(specularity, cos_theta);
    double Tr = 1 - Re;
    double reflection_probability = Re;
    bool is_reflection = XORShift::frand() < reflection_probability;
    return (emission + (is_reflection ? // Russian roulette between reflection and refraction
        integrate_radiance(reflection_ray, depth) :
        (albedo * integrate_radiance(Ray(x, tdir), depth)))) * recip_surface_interaction_probability;
}

void accumulate_radiance(int w, int h, RGB *const backbuffer, int& accumulations) {

    Ray cam(Vector3d(50, 52, 285), normalize(Vector3d(0, -0.042612, -1))); // cam pos, dir

    ++accumulations;
    float blendFactor = 1.0f / accumulations;

    Vector3d cx = Vector3d(w*.5135 / h, 0, 0);
    Vector3d cy = normalize(cross(cx, cam.direction))*.5135;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < h; y++) {
        for (unsigned short x = 0; x < w; x++) {
            // Stratify samples in 2x2 in image plane.
            int sx = accumulations % 2;
            int sy = (accumulations >> 1) % 2;

            double r1 = 2 * XORShift::frand(), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
            double r2 = 2 * XORShift::frand(), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
            Vector3d d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) +
                cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.direction;
            RGB r = integrate_radiance(Ray(cam.origin + d * 140, normalize(d)), 0);
            // Camera rays are pushed ^^^^^ forward to start in interior
            int i = (h - y - 1) * w + x;
            backbuffer[i] = lerp(backbuffer[i], r, blendFactor);
        }
    }
}

}

#endif // _SMALL_VPT_H_