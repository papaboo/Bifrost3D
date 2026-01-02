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
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>

using namespace Bifrost::Math;

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
}

struct Ray { Vector3d o, d; Ray() {} Ray(Vector3d o_, Vector3d d_) : o(o_), d(d_) {} };
enum class BSDF { Diffuse, Specular, Glass };

struct Sphere {
	double radius;
    Vector3d position;
    RGB emission, albedo;
    BSDF bsdf;
	Sphere(double r, Vector3d p, RGB e, RGB a, BSDF bsdf):
        radius(r), position(p), emission(e), albedo(a), bsdf(bsdf) {}
	double intersect(const Ray &r, double *tin = NULL, double *tout = NULL) const { // returns distance, 0 if nohit
        Vector3d op = position - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
		double t, eps=1e-4, b=dot(op, r.d), det=b*b-dot(op, op)+ radius * radius;
		if (det<0) return 0; else det=sqrt(det);
		if (tin && tout) {*tin=(b-det<=0)?0:b-det;*tout=b+det;}
		return (t=b-det)>eps ? t : ((t=b+det)>eps ? t : 0);
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

Sphere homogeneousMedium(300, Vector3d(50,50,80), RGB::black(), RGB::black(), BSDF::Diffuse);
const double sigma_s = 0.009, sigma_a = 0.006, sigma_t = sigma_s+sigma_a;
inline bool intersect(const Ray &r, double &t, int &id, double tmax=1e20){
	double n=sizeof(spheres)/sizeof(Sphere), d, inf=t=tmax;
	for(int i=int(n);i--;) if((d=spheres[i].intersect(r))&&d<t){t=d;id=i;}
	return t<inf;
}
inline double sampleSegment(double epsilon, float sigma, float smax) {
	return -log(1.0 - epsilon * (1.0 - exp(-sigma * smax))) / sigma;
}
inline Vector3d sampleHG(double g, double e1, double e2) {
	//double s=2.0*e1-1.0, f = (1.0-g*g)/(1.0+g*s), cost = 0.5*(1.0/g)*(1.0+g*g-f*f), sint = sqrt(1.0-cost*cost);
	double s = 1.0-2.0*e1, cost = (s + 2.0*g*g*g * (-1.0 + e1) * e1 + g*g*s + 2.0*g*(1.0 - e1+e1*e1))/((1.0+g*s)*(1.0+g*s)), sint = sqrt(1.0-cost*cost);
	return Vector3d(cos(2.0 * M_PI * e2) * sint, sin(2.0 * M_PI * e2) * sint, cost);
}

inline double scatter(const Ray &r, Ray *sRay, double tin, float tout, double &s) {
	s = sampleSegment(XORShift::frand(), float(sigma_s), float(tout - tin));
	Vector3d x = r.o + r.d *tin + r.d * s;
	Vector3d dir = sampleHG(-0.5,XORShift::frand(), XORShift::frand()); //Sample a direction ~ Henyey-Greenstein's phase function
	Vector3d u, v;
    compute_tangents(r.d, u, v);
	dir = u*dir.x+v*dir.y+r.d*dir.z;
	if (sRay)	*sRay = Ray(x, dir);
	return (1.0 - exp(-sigma_s * (tout - tin)));
}
RGB radiance(const Ray &r, int depth) {
    // Avoid stack overflow from recursion
    if (depth > 250)
        return RGB::black();

	double t;                               // distance to intersection
	int id=0;                               // id of intersected object
	double tnear, tfar, scaleBy=1.0, absorption=1.0;
	bool intrsctmd = homogeneousMedium.intersect(r, &tnear, &tfar) > 0;
	if (intrsctmd) {
		Ray sRay;
		double s, ms = scatter(r, &sRay, tnear, tfar, s), prob_s = ms;
		scaleBy = 1.0/(1.0-prob_s);
		if (XORShift::frand() <= prob_s) { // Sample surface or volume?
			if (!intersect(r, t, id, tnear + s))
				return radiance(sRay, ++depth) * ms * (1.0/prob_s);
			scaleBy = 1.0;
		}
		else
			if (!intersect(r, t, id)) return RGB::black();
		if (t >= tnear) {
			double dist = (t > tfar ? tfar - tnear : t - tnear); 
			absorption=exp(-sigma_t * dist);
		}
	}
	else
		if (!intersect(r, t, id)) return RGB::black();
	const Sphere &obj = spheres[id];        // the hit object
    Vector3d x = r.o + r.d*t, n = normalize(x - obj.position), nl = dot(n, r.d) < 0 ? n : n * -1;
    RGB f = obj.albedo, Le = obj.emission;
	double p = f.r>f.g && f.r>f.b ? f.r : f.g>f.b ? f.g : f.b; // max refl
	if (++depth>5) if (XORShift::frand()<p) {f=f*(1/p);} else return RGB::black(); // Russian roulette
	if (dot(n, nl)>0 || obj.bsdf != BSDF::Glass) {f = f * absorption; Le = obj.emission * absorption;} // no absorption inside glass
	else scaleBy=1.0;
	if (obj.bsdf == BSDF::Diffuse) {
		double r1=2*M_PI*XORShift::frand(), r2=XORShift::frand(), r2s=sqrt(r2);
        Vector3d w = nl;
        Vector3d u = normalize(cross((fabs(w.x) > .1 ? Vector3d(0, 1, 0) : Vector3d(1, 0, 0)), w));
        Vector3d v = cross(w, u);
		Vector3d d = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2));
		return (Le + f * radiance(Ray(x,d),depth)) * scaleBy;
	} else if (obj.bsdf == BSDF::Specular)
		return (Le + f * radiance(Ray(x,r.d-n*2*dot(n, r.d)),depth)) * scaleBy;
	Ray reflRay(x, r.d-n*2*dot(n, r.d));     // Ideal dielectric REFRACTION
	bool into = dot(n, nl)>0;                // Ray from outside going in?
	double nc=1, nt=1.5, nnt=into?nc/nt:nt/nc, ddn=dot(r.d, nl), cos2t;
	if ((cos2t=1-nnt*nnt*(1-ddn*ddn))<0)    // Total internal reflection
		return (Le + f * radiance(reflRay,depth));
    Vector3d tdir = normalize(r.d*nnt - n*((into?1:-1)*(ddn*nnt+sqrt(cos2t))));
	double a=nt-nc, b=nt+nc, R0=a*a/(b*b), c = 1-(into?-ddn:dot(n, tdir));
	double Re=R0+(1-R0)*c*c*c*c*c,Tr=1-Re,P=.25+.5*Re,RP=Re/P,TP=Tr/(1-P);
		return (Le + (depth>2 ? (XORShift::frand()<P ?   // Russian roulette
		radiance(reflRay,depth)*RP:(f * radiance(Ray(x,tdir),depth)*TP)) :
	    radiance(reflRay,depth)*Re+(f * radiance(Ray(x,tdir),depth)*Tr)))*scaleBy;
}

void smallvpt_accumulateRadiance(int w, int h, RGB *const backbuffer, int& accumulations) {

    Ray cam(Vector3d(50, 52, 285), normalize(Vector3d(0, -0.042612, -1))); // cam pos, dir

    ++accumulations;
    float blendFactor = 1.0f / accumulations;

    Vector3d cx = Vector3d(w*.5135 / h, 0, 0);
    Vector3d cy = normalize(cross(cx, cam.d))*.5135;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < h; y++) {
        for (unsigned short x = 0; x < w; x++) {
            // Stratify samples in 2x2 in image plane.
            int sx = accumulations % 2;
            int sy = (accumulations >> 1) % 2;

            double r1 = 2 * XORShift::frand(), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
            double r2 = 2 * XORShift::frand(), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
            Vector3d d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) +
                cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.d;
            RGB r = radiance(Ray(cam.o + d * 140, normalize(d)), 0);
            // Camera rays are pushed ^^^^^ forward to start in interior
            int i = (h - y - 1) * w + x;
            backbuffer[i] = lerp(backbuffer[i], r, blendFactor);
        }
    }
}

#endif // _SMALL_VPT_H_