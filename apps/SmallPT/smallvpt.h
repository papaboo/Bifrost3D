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

struct Vec {        // Usage: time ./smallpt 5000 && xv image.ppm
	double x, y, z;                  // position, also color (r,g,b)
	Vec(double x_=0, double y_=0, double z_=0){ x=x_; y=y_; z=z_; }
	Vec operator+(const Vec &b) const { return Vec(x+b.x,y+b.y,z+b.z); }
	Vec operator-(const Vec &b) const { return Vec(x-b.x,y-b.y,z-b.z); }
	Vec operator*(double b) const { return Vec(x*b,y*b,z*b); }
	Vec mult(const Vec &b) const { return Vec(x*b.x,y*b.y,z*b.z); }
	Vec& norm1(){ return *this = *this * (1/sqrt(x*x+y*y+z*z)); }
	float length() {return sqrt(x*x+y*y+z*z); }
	// double dot(const Vec &b) const { return x*b.x+y*b.y+z*b.z; } // cross:
	// Vec operator%(Vec&b){return Vec(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);}
};
__always_inline__ Vec cross(Vec lhs, Vec rhs) {
    return Vec((lhs.y * rhs.z) - (lhs.z * rhs.y),
               (lhs.z * rhs.x) - (lhs.x * rhs.z),
               (lhs.x * rhs.y) - (lhs.y * rhs.x));
}
double dot(Vec lhs, Vec rhs) { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z; }
Vec normalize(Vec v) { return v.norm1(); }


struct Ray { Vec o, d; Ray() {} Ray(Vec o_, Vec d_) : o(o_), d(d_) {} };
enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()

struct Sphere {
	double rad;       // radius
    Vec p;
    RGB e, c;      // position, emission, color
	Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)
	Sphere(double rad_, Vec p_, RGB e_, RGB c_, Refl_t refl_):
	rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}
	double intersect(const Ray &r, double *tin = NULL, double *tout = NULL) const { // returns distance, 0 if nohit
		Vec op = p-r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
		double t, eps=1e-4, b=dot(op, r.d), det=b*b-dot(op, op)+rad*rad;
		if (det<0) return 0; else det=sqrt(det);
		if (tin && tout) {*tin=(b-det<=0)?0:b-det;*tout=b+det;}
		return (t=b-det)>eps ? t : ((t=b+det)>eps ? t : 0);
	}
};
//Sphere spheres[] = {//Scene: radius, position, emission, color, material 
//	Sphere(26.5,Vec(27,18.5,78),       Vec(),Vec(1,1,1)*.75, SPEC),//Mirr
//	Sphere(12,Vec(70,43,78),       Vec(),Vec(0.27,0.8,0.8), REFR),//Glas
//	Sphere(8, Vec(55,87,78),Vec(),  Vec(1,1,1)*.75, DIFF), //Lite
//	Sphere(4, Vec(55,80,78),Vec(10,10,10),  Vec(), DIFF) //Lite
//};

Sphere spheres[] = { // Scene: radius, position, emission, color, material
    Sphere(1e5, Vec(1e5 + 1, 40.8, 81.6), RGB::black(), RGB(.75f, .25f, .25f), DIFF),//Left
    Sphere(1e5, Vec(-1e5 + 99, 40.8, 81.6), RGB::black(), RGB(.25f, .25f, .75f), DIFF),//Rght
    Sphere(1e5, Vec(50, 40.8, 1e5), RGB::black(), RGB(.75f), DIFF),//Back
    Sphere(1e5, Vec(50, 40.8, -1e5 + 170), RGB::black(), RGB::black(), DIFF),//Frnt
    Sphere(1e5, Vec(50, 1e5, 81.6), RGB::black(), RGB(.75f), DIFF),//Botm
    Sphere(1e5, Vec(50, -1e5 + 81.6, 81.6), RGB::black(), RGB(.75f), DIFF),//Top
    Sphere(16.5, Vec(27, 16.5, 47), RGB::black(), RGB(.999f), SPEC),//Mirr
    Sphere(16.5, Vec(73, 16.5, 78), RGB::black(), RGB(.999f), REFR),//Glas
    Sphere(600, Vec(50, 681.6 - .27, 81.6), RGB(12.0f), RGB::black(), DIFF) //Light
};

Sphere homogeneousMedium(300, Vec(50,50,80), RGB::black(), RGB::black(), DIFF);
const double sigma_s = 0.009, sigma_a = 0.006, sigma_t = sigma_s+sigma_a;
inline bool intersect(const Ray &r, double &t, int &id, double tmax=1e20){
	double n=sizeof(spheres)/sizeof(Sphere), d, inf=t=tmax;
	for(int i=int(n);i--;) if((d=spheres[i].intersect(r))&&d<t){t=d;id=i;}
	return t<inf;
}
inline double sampleSegment(double epsilon, float sigma, float smax) {
	return -log(1.0 - epsilon * (1.0 - exp(-sigma * smax))) / sigma;
}
inline Vec sampleHG(double g, double e1, double e2) {
	//double s=2.0*e1-1.0, f = (1.0-g*g)/(1.0+g*s), cost = 0.5*(1.0/g)*(1.0+g*g-f*f), sint = sqrt(1.0-cost*cost);
	double s = 1.0-2.0*e1, cost = (s + 2.0*g*g*g * (-1.0 + e1) * e1 + g*g*s + 2.0*g*(1.0 - e1+e1*e1))/((1.0+g*s)*(1.0+g*s)), sint = sqrt(1.0-cost*cost);
	return Vec(cos(2.0 * M_PI * e2) * sint, sin(2.0 * M_PI * e2) * sint, cost);
}
inline void generateOrthoBasis(Vec &u, Vec &v, Vec w) {
	Vec coVec = w;
	if (fabs(w.x) <= fabs(w.y))
		if (fabs(w.x) <= fabs(w.z)) coVec = Vec(0,-w.z,w.y);
		else coVec = Vec(-w.y,w.x,0);
	else if (fabs(w.y) <= fabs(w.z)) coVec = Vec(-w.z,0,w.x);
	else coVec = Vec(-w.y,w.x,0);
	coVec.norm();
	u = cross(w, coVec),
	v = cross(w, u);
}
inline double scatter(const Ray &r, Ray *sRay, double tin, float tout, double &s) {
	s = sampleSegment(XORShift::frand(), float(sigma_s), float(tout - tin));
	Vec x = r.o + r.d *tin + r.d * s;
	Vec dir = sampleHG(-0.5,XORShift::frand(), XORShift::frand()); //Sample a direction ~ Henyey-Greenstein's phase function
	Vec u,v;
	generateOrthoBasis(u,v,r.d);
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
		if (XORShift::frand() <= prob_s) {// Sample surface or volume?
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
    Vec x = r.o + r.d*t, n = (x - obj.p).norm(), nl = dot(n, r.d) < 0 ? n : n * -1;
    RGB f = obj.c, Le = obj.e;
	double p = f.r>f.g && f.r>f.b ? f.r : f.g>f.b ? f.g : f.b; // max refl
	if (++depth>5) if (XORShift::frand()<p) {f=f*(1/p);} else return RGB::black(); //R.R.
	if (dot(n, nl)>0 || obj.refl != REFR) {f = f * absorption; Le = obj.e * absorption;}// no absorption inside glass
	else scaleBy=1.0;
	if (obj.refl == DIFF) {                  // Ideal DIFFUSE reflection
		double r1=2*M_PI*XORShift::frand(), r2=XORShift::frand(), r2s=sqrt(r2);
        Vec w = nl;
        Vec u = cross((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)), w).norm();
        Vec v = cross(w, u);
		Vec d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2)).norm();
		return (Le + f * radiance(Ray(x,d),depth)) * scaleBy;
	} else if (obj.refl == SPEC)            // Ideal SPECULAR reflection
		return (Le + f * radiance(Ray(x,r.d-n*2*dot(n, r.d)),depth)) * scaleBy;
	Ray reflRay(x, r.d-n*2*dot(n, r.d));     // Ideal dielectric REFRACTION
	bool into = dot(n, nl)>0;                // Ray from outside going in?
	double nc=1, nt=1.5, nnt=into?nc/nt:nt/nc, ddn=dot(r.d, nl), cos2t;
	if ((cos2t=1-nnt*nnt*(1-ddn*ddn))<0)    // Total internal reflection
		return (Le + f * radiance(reflRay,depth));
	Vec tdir = (r.d*nnt - n*((into?1:-1)*(ddn*nnt+sqrt(cos2t)))).norm();
	double a=nt-nc, b=nt+nc, R0=a*a/(b*b), c = 1-(into?-ddn:dot(n, tdir));
	double Re=R0+(1-R0)*c*c*c*c*c,Tr=1-Re,P=.25+.5*Re,RP=Re/P,TP=Tr/(1-P);
		return (Le + (depth>2 ? (XORShift::frand()<P ?   // Russian roulette
		radiance(reflRay,depth)*RP:(f * radiance(Ray(x,tdir),depth)*TP)) :
	    radiance(reflRay,depth)*Re+(f * radiance(Ray(x,tdir),depth)*Tr)))*scaleBy;
}

void smallvpt_accumulateRadiance(int w, int h, RGB *const backbuffer, int& accumulations) {

    Ray cam(Vec(50, 52, 285), Vec(0, -0.042612, -1).norm()); // cam pos, dir

    ++accumulations;
    float blendFactor = 1.0f / accumulations;

    Vec cx = Vec(w*.5135 / h);
    Vec cy = cross(cx, cam.d).norm()*.5135;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < h; y++) {
        for (unsigned short x = 0; x < w; x++) {
            // Stratify samples in 2x2 in image plane.
            int sx = accumulations % 2;
            int sy = (accumulations >> 1) % 2;

            double r1 = 2 * XORShift::frand(), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
            double r2 = 2 * XORShift::frand(), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
            Vec d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) +
                cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.d;
            RGB r = radiance(Ray(cam.o + d * 140, d.norm()), 0);
            // Camera rays are pushed ^^^^^ forward to start in interior
            int i = (h - y - 1) * w + x;
            backbuffer[i] = lerp(backbuffer[i], r, blendFactor);
        }
    }
}

#endif // _SMALL_VPT_H_