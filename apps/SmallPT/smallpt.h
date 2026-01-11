// Smallpt
// ----------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _SMALL_PT_H_
#define _SMALL_PT_H_

#include <Bifrost/Math/Color.h>
#include <Bifrost/Math/Constants.h>
#include <Bifrost/Math/RNG.h>
#include <Bifrost/Math/Utils.h>
#include <Bifrost/Math/Vector.h>

#include <math.h>

namespace smallpt {

using namespace Bifrost::Math;

struct Ray { 
    Vector3d origin, direction; 
    Ray(Vector3d o, Vector3d d)
        : origin(o), direction(d) {}
};

enum class BSDF { Diffuse, Specular, Glass };

struct Sphere {
    double radius;
    Vector3d position;
    RGB emission, color;
    BSDF bsdf;
    Sphere(double r, Vector3d p, RGB e, RGB c, BSDF b)
        : radius(r), position(p), emission(e), color(c), bsdf(b) {}

    double intersect(Ray r) const { // returns distance, 0 if nohit
        Vector3d op = position - r.origin; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        double t, eps = 1e-4, b = dot(op, r.direction), det = b*b - magnitude_squared(op) + radius*radius;
        if (det<0) return 0; else det = sqrt(det);
        return (t = b - det)>eps ? t : ((t = b + det)>eps ? t : 0);
    }
};
Sphere scene[] = { // Scene: radius, position, emission, color, material
    Sphere(1e5, Vector3d(1e5 + 1, 40.8, 81.6), RGB::black(), RGB(.75f, .25f, .25f), BSDF::Diffuse),//Left
    Sphere(1e5, Vector3d(-1e5 + 99, 40.8, 81.6), RGB::black(), RGB(.25f, .25f, .75f), BSDF::Diffuse),//Rght
    Sphere(1e5, Vector3d(50, 40.8, 1e5), RGB::black(), RGB(.75f), BSDF::Diffuse),//Back
    Sphere(1e5, Vector3d(50, 40.8, -1e5 + 170), RGB::black(), RGB::black(), BSDF::Diffuse),//Frnt
    Sphere(1e5, Vector3d(50, 1e5, 81.6), RGB::black(), RGB(.75f), BSDF::Diffuse),//Botm
    Sphere(1e5, Vector3d(50, -1e5 + 81.6, 81.6), RGB::black(), RGB(.75f), BSDF::Diffuse),//Top
    Sphere(16.5, Vector3d(27, 16.5, 47), RGB::black(), RGB(.999f), BSDF::Specular),//Mirr
    Sphere(16.5, Vector3d(73, 16.5, 78), RGB::black(), RGB(.999f), BSDF::Glass),//Glas
    Sphere(600, Vector3d(50, 681.6 - .27, 81.6), RGB(12.0f), RGB::black(), BSDF::Diffuse) //Light
};

inline bool intersect(Ray r, double &t, int &id) {
    double n = sizeof(scene) / sizeof(Sphere), d, inf = t = 1e20;
    for (int i = int(n); i--;) if ((d = scene[i].intersect(r)) && d<t){ t = d; id = i; }
    return t < inf;
}

RGB radiance(const Ray &ray, int depth, RNG::LinearCongruential& rng) {
    // distance to intersection
    double t;
    // id of intersected object
    int id = 0;
    // if miss, return black  
    if (depth > 20 || !intersect(ray, t, id)) return RGB::black();
    // the hit object
    const Sphere &obj = scene[id];
    Vector3d pos = ray.origin + ray.direction * t;
    Vector3d norm = normalize(pos - obj.position);
    Vector3d nl = dot(norm, ray.direction) < 0 ? norm : norm*-1;
    RGB f = obj.color;
    float maxRefl = f.r>f.g && f.r>f.b ? f.r : f.g>f.b ? f.g : f.b;
    if (++depth > 5)
        if (rng.sample1f() < maxRefl) f = f * (1 / maxRefl);
        else return obj.emission; // Russion roulette

        if (obj.bsdf == BSDF::Diffuse) {
            double r1 = 2.0f * PI<float>() * rng.sample1f();
            double r2 = rng.sample1f();
            double r2s = sqrt(r2);
            // Tangent space
            Vector3d w = nl;
            Vector3d u = normalize(cross(fabs(w.x) > 0.1 ? Vector3d(0, 1, 0) : Vector3d(1, 0, 0), w));
            Vector3d v = cross(w, u);
            Vector3d dir = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2));
            return obj.emission + f * radiance(Ray(pos, dir), depth, rng);
        }
        else if (obj.bsdf == BSDF::Specular) {
            Vector3d reflect = ray.direction - nl * 2 * dot(nl, ray.direction);
            return obj.emission + f * radiance(Ray(pos, reflect), depth, rng);
        }
        else { // Ideal dielectric refraction, glass.
            Ray reflRay(pos, ray.direction - norm * 2 * dot(norm, ray.direction));
            bool into = dot(norm, nl) > 0; // Ray from outside going in?
            const float nc = 1, nt = 1.5;
            double nnt = into ? nc / nt : nt / nc, ddn = dot(ray.direction, nl), cos2t;
            if ((cos2t = 1 - nnt*nnt*(1 - ddn*ddn))<0)    // Total internal reflection
                return obj.emission + f * radiance(reflRay, depth, rng);
            Vector3d tdir = normalize(ray.direction*nnt - norm*((into ? 1 : -1)*(ddn*nnt + sqrt(cos2t))));
            float a = nt - nc, b = nt + nc;
            float R0 = a*a / (b*b);
            float c = 1.0f - float(into ? -ddn : dot(tdir, norm)); // cosTheta
            float Re = R0 + (1.0f - R0)*c*c*c*c*c; // Schlick's fresnel approximation.
            float Tr = 1.0f - Re;
            float P = .25f + .5f * Re;
            float RP = Re / P;
            float TP = Tr / (1.0f - P);
            return obj.emission + f * (depth>2 ? (rng.sample1f()<P ?   // Russian roulette
                radiance(reflRay, depth, rng)*RP : radiance(Ray(pos, tdir), depth, rng)*TP) :
                radiance(reflRay, depth, rng)*Re + radiance(Ray(pos, tdir), depth, rng)*Tr);
        }
}

void accumulate_radiance(int w, int h, RGB *const backbuffer, int& accumulations) {

    Ray cam(Vector3d(50, 52, 295.6), normalize(Vector3d(0, -0.042612, -1)));

    ++accumulations;
    float blendFactor = 1.0f / accumulations;

    Vector3d cx = Vector3d(w * 0.5135 / h, 0, 0), cy = normalize(cross(cx, cam.direction)) * 0.5135;

    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < h; ++y) {
        for (unsigned short x = 0; x < w; ++x) {
            // Stratify samples in 2x2 in image plane.
            int sx = accumulations % 2;
            int sy = (accumulations >> 1) % 2;
            int index = (y * 2 + sy) * (w * 2) + x * 2 + sx;
            RNG::LinearCongruential rng = RNG::LinearCongruential(RNG::jenkins_hash(unsigned int(index)) ^ reverse_bits(unsigned int(accumulations)));
            double r1 = 2 * rng.sample1f(), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
            double r2 = 2 * rng.sample1f(), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
            Vector3d d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) +
                cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.direction;
            RGB r = radiance(Ray(cam.origin + d * 140, normalize(d)), 0, rng);
            //            Camera rays are pushed ^^^^^ forward to start in interior
            int i = y * w + x;
            backbuffer[i] = lerp(backbuffer[i], r, blendFactor);
        }
    }
}

} // NS smallpt

#endif // _SMALL_PT_H_