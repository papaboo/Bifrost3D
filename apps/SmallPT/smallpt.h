// Smallpt
// ----------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _SMALL_PT_H_
#define _SMALL_PT_H_

#include <Core/Array.h>
#include <Math/Color.h>
#include <Math/Constants.h>
#include <Math/Vector.h>
#include <Math/Utils.h>

#include <math.h>

namespace smallpt {

using namespace Cogwheel::Math;

class LinearCongruential {
private:
    static const unsigned int multiplier = 1664525u;
    static const unsigned int increment = 1013904223u;
    static const unsigned int max = 0xFFFFFFFFu; // uint32 max.

    unsigned int mState;

    unsigned int next() {
        mState = multiplier * mState + increment;
        return mState;
    }

    float nextFloat() {
        const float invMax = 1.0f / (float(max) + 1.0f);
        return float(next()) * invMax;
    }

public:
    LinearCongruential(unsigned int seed)
        : mState(seed) { }

    unsigned int getSeed() const { return mState; }

    float sample1D() {
        return nextFloat();
    }
};

// Robert Jenkins hash function.
// https://gist.github.com/badboy/6267743
inline unsigned int RobertJenkinsHash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// Divide and conquor bit reversal.
// https://graphics.stanford.edu/~seander/bithacks.html#ReverseParallel
inline unsigned int reverseBits(unsigned int v) {
    v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
    v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
    v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
    v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
    return (v >> 16) | (v << 16);
}

struct Ray { 
    Vector3d origin, direction; 
    Ray(Vector3d o, Vector3d d)
        : origin(o), direction(d) {}
};

enum class BSDF { Diffuse, Specular, Glass };

struct Sphere { // TODO Align to float4?
    double radius;
    Vector3d position;
    RGB emission, color;
    BSDF bsdf;
    Sphere(double r, Vector3d p, RGB e, RGB c, BSDF b)
        : radius(r), position(p), emission(e), color(c), bsdf(b) {}

    double intersect(Ray r) const { // returns distance, 0 if nohit
        Vector3d op = position - r.origin; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        double t, eps = 1e-4, b = dot(op, r.direction), det = b*b - squaredMagnitude(op) + radius*radius;
        if (det<0) return 0; else det = sqrt(det);
        return (t = b - det)>eps ? t : ((t = b + det)>eps ? t : 0);
    }
};
Sphere scene[] = { // Scene: radius, position, emission, color, material
    Sphere(1e5, Vector3d(1e5 + 1, 40.8, 81.6), RGB::black(), RGB(.75f, .25f, .25f), BSDF::Diffuse),//Left
    Sphere(1e5, Vector3d(-1e5 + 99, 40.8, 81.6), RGB::black(), RGB(.25f, .25f, .75f), BSDF::Diffuse),//Rght
    Sphere(1e5, Vector3d(50, 40.8, 1e5), RGB::black(), RGB(.75f, .75f, .75f), BSDF::Diffuse),//Back
    Sphere(1e5, Vector3d(50, 40.8, -1e5 + 170), RGB(), RGB::black(), BSDF::Diffuse),//Frnt
    Sphere(1e5, Vector3d(50, 1e5, 81.6), RGB::black(), RGB(.75f, .75f, .75f), BSDF::Diffuse),//Botm
    Sphere(1e5, Vector3d(50, -1e5 + 81.6, 81.6), RGB::black(), RGB(.75f, .75f, .75f), BSDF::Diffuse),//Top
    Sphere(16.5, Vector3d(27, 16.5, 47), RGB::black(), RGB(1.0f, 1.0f, 1.0f)*.999f, BSDF::Specular),//Mirr
    Sphere(16.5, Vector3d(73, 16.5, 78), RGB::black(), RGB(1.0f, 1.0f, 1.0f)*.999f, BSDF::Glass),//Glas
    Sphere(600, Vector3d(50, 681.6 - .27, 81.6), RGB(12.0f, 12.0f, 12.0f), RGB::black(), BSDF::Diffuse) //Lite
};

inline bool intersect(Ray r, double &t, int &id) {
    double n = sizeof(scene) / sizeof(Sphere), d, inf = t = 1e20;
    for (int i = int(n); i--;) if ((d = scene[i].intersect(r)) && d<t){ t = d; id = i; }
    return t < inf;
}

RGB radiance(const Ray &ray, int depth, LinearCongruential& rng) {
    // TODO AVH Return intersection struct with a t and a sphere reference. Wrap it in an option.
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
        if (rng.sample1D() < maxRefl) f = f * (1 / maxRefl);
        else return obj.emission; // Russion roulette

        if (obj.bsdf == BSDF::Diffuse) {
            double r1 = 2.0f * PI<float>() * rng.sample1D();
            double r2 = rng.sample1D();
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
            // TODO AVH investigate this probability and switch to always use russian roulette. Then we could switch from recursive to iterative ray tracing.
            float P = .25f + .5f * Re;
            float RP = Re / P;
            float TP = Tr / (1.0f - P);
            return obj.emission + f * (depth>2 ? (rng.sample1D()<P ?   // Russian roulette
                radiance(reflRay, depth, rng)*RP : radiance(Ray(pos, tdir), depth, rng)*TP) :
                radiance(reflRay, depth, rng)*Re + radiance(Ray(pos, tdir), depth, rng)*Tr);
        }
}

void accumulateRadiance(Ray cam, int w, int h,
                        RGB *const backbuffer, int& accumulations) {

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
            LinearCongruential rng = LinearCongruential(RobertJenkinsHash(index) ^ reverseBits(accumulations));
            double r1 = 2 * rng.sample1D(), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
            double r2 = 2 * rng.sample1D(), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
            Vector3d d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) +
                cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.direction;
            int i = (h - y - 1) * w + x;
            RGB r = radiance(Ray(cam.origin + d * 140, normalize(d)), 0, rng);
            //            Camera rays are pushed ^^^^^ forward to start in interior
            backbuffer[i] = lerp(backbuffer[i], r, blendFactor);
        }
    }
}

} // NS smallpt

#endif // _SMALL_PT_H_