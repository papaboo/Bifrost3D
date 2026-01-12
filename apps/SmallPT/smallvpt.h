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
#include <Bifrost/Math/Distributions.h>
#include <Bifrost/Math/RNG.h>
#include <Bifrost/Math/Utils.h>
#include <Bifrost/Math/Vector.h>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace Bifrost::Math;

namespace smallvpt {

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

Sphere homogeneous_medium(300, Vector3d(50, 50, 80), RGB::black(), RGB(0.9f, 0.6f, 0.3f), BSDF::Diffuse);
const float sigma_t = 0.01f; // Attenuation coefficient

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

RGB integrate_radiance(const Ray &ray, int depth, RNG::LinearCongruential rng) {
    // Avoid stack overflow from recursion
    if (depth > 250)
        return RGB::black();

    double medium_t_near, medium_t_far;
    bool intersects_medium = homogeneous_medium.intersect(ray, &medium_t_near, &medium_t_far) > 0;
    float scatter_t = 1e20f;
    if (intersects_medium)
        scatter_t = medium_t_near + Distributions::Exponential::sample_distance(sigma_t, rng.sample1f());

    double surface_t;
    int scene_object_id = -1;
    bool surface_intersection = intersect_scene(ray, surface_t, scene_object_id, scatter_t);

    if (!surface_intersection && !intersects_medium)
        // No intersection
        return RGB::black();
    else if (scatter_t <= surface_t) {
        // Scattering event before surface interaction

        // Russian roulette based on single scattering albedo, which is the absorption probability.
        RGB single_scattering_albedo = homogeneous_medium.albedo;
        if (rng.sample1f() >= average(single_scattering_albedo))
            return RGB::black();

        // Create new ray from scattering event.
        Vector3d scattering_position = ray.origin + ray.direction * scatter_t;
        Vector3f direction_sample = Distributions::HenyeyGreenstein::sample_direction(-0.5, Vector3f(ray.direction), rng.sample2f());
        Ray scattered_ray = Ray(scattering_position, Vector3d(direction_sample));

        return single_scattering_albedo * integrate_radiance(scattered_ray, ++depth, rng);
    } else {
        // Surface interaction before scattering event.
        const Sphere& scene_object = spheres[scene_object_id];
        Vector3d x = ray.origin + ray.direction * surface_t;
        Vector3d n = normalize(x - scene_object.position);
        Vector3d nl = dot(n, ray.direction) < 0 ? n : n * -1;
        RGB albedo = scene_object.albedo, emission = scene_object.emission;

        // Russian roulette after 5 interactions.
        // The decision is taken based on local albedo information.
        // If the ray passes, then albedo is scaled by the russian roulette probability and tracing continues.
        // The material's emission is always applied and therefore not scaled by russian roulette probability.
        if (++depth > 5) {
            float max_albedo_value = max(albedo.r, max(albedo.g, albedo.b));
            if (rng.sample1f() < max_albedo_value)
                albedo *= 1.0f / max_albedo_value;
            else
                return emission;
        }

        if (scene_object.bsdf == BSDF::Diffuse) {
            float r1 = 2 * float(M_PI) * rng.sample1f();
            float r2 = rng.sample1f();
            float r2s = sqrt(r2);
            Vector3f normal = Vector3f(nl);
            Vector3f tangent, bitangent;
            compute_tangents(normal, tangent, bitangent);
            Vector3d d = (Vector3d)normalize(tangent * cosf(r1) * r2s + bitangent * sinf(r1) * r2s + normal * sqrt(1 - r2));
            return (emission + albedo * integrate_radiance(Ray(x, d), depth, rng));
        }

        Ray reflection_ray(x, reflect(ray.direction, n));
        if (scene_object.bsdf == BSDF::Specular)
            return (emission + albedo * integrate_radiance(reflection_ray, depth, rng));

        // Glass
        bool into = dot(n, nl) > 0; // Ray from outside going in?
        double air_ior = 1;
        double glass_ior = 1.5;
        double relative_ior = into ? air_ior / glass_ior : glass_ior / air_ior;
        double ddn = dot(ray.direction, nl);
        double cos2t;
        if ((cos2t = 1 - relative_ior * relative_ior * (1 - ddn * ddn)) < 0)    // Total internal reflection
            return (emission + integrate_radiance(reflection_ray, depth, rng));
        Vector3d tdir = normalize(ray.direction * relative_ior - n * ((into ? 1 : -1)*(ddn * relative_ior + sqrt(cos2t))));
        double specularity = dielectric_specularity(air_ior, glass_ior);
        double cos_theta = into ? -ddn : dot(n, tdir);
        double Re = schlick_fresnel(specularity, cos_theta);
        double Tr = 1 - Re;
        double reflection_probability = Re;
        bool is_reflection = rng.sample1f() < reflection_probability;
        return (emission + (is_reflection ? // Russian roulette between reflection and refraction
            integrate_radiance(reflection_ray, depth, rng) :
            (albedo * integrate_radiance(Ray(x, tdir), depth, rng))));
    }
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
            int index = y * w + x;

            RNG::LinearCongruential rng = RNG::LinearCongruential(RNG::jenkins_hash(unsigned int(index)) ^ reverse_bits(unsigned int(accumulations)));

            double r1 = 2 * rng.sample1f(), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
            double r2 = 2 * rng.sample1f(), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
            Vector3d d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) +
                cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.direction;
            RGB r = integrate_radiance(Ray(cam.origin + d * 140, normalize(d)), 0, rng);
            // Camera rays are pushed ^^^^^ forward to start in interior
            backbuffer[index] = lerp(backbuffer[index], r, blendFactor);
        }
    }
}

}

#endif // _SMALL_VPT_H_