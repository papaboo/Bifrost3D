// OptiX sphere intersection program.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------
// Inspired by the OptiX samples.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Shading/Utils.h>

#include <optix.h>
#include <optixu/optixu_aabb.h>
#include <optixu/optixu_math.h>

using namespace optix;

rtDeclareVariable(float4, sphere, , );

rtDeclareVariable(Ray, ray, rtCurrentRay, );

rtDeclareVariable(float2, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

// Intersection copied from CUDA OptiX Sampels.
template<bool use_robust_method>
__inline_dev__ void intersect_sphere() {
    float3 center = make_float3(sphere);
    float3 O = ray.origin - center;
    float3 D = ray.direction;
    float radius = sphere.w;

    float b = dot(O, D);
    float c = dot(O, O) - radius*radius;
    float disc = b*b - c;
    if (disc > 0.0f){
        float sdisc = sqrtf(disc);
        float root1 = (-b - sdisc);

        bool do_refine = false;

        float root11 = 0.0f;

        if (use_robust_method && fabsf(root1) > 10.f * radius)
            do_refine = true;

        if (do_refine) {
            // refine root1
            float3 O1 = O + root1 * ray.direction;
            b = dot(O1, D);
            c = dot(O1, O1) - radius*radius;
            disc = b*b - c;

            if (disc > 0.0f) {
                sdisc = sqrtf(disc);
                root11 = (-b - sdisc);
            }
        }

        bool check_second = true;
        if (rtPotentialIntersection(root1 + root11)) {
            shading_normal = geometric_normal = (O + (root1 + root11)*D) / radius;
            float s = (shading_normal.z + 1.0f) * 0.5f;
            float t = (atan2(shading_normal.y, shading_normal.x) + PIf) * 0.5f / PIf;
            texcoord = make_float2(s, t);
            if (rtReportIntersection(0))
                check_second = false;
        }
        if (check_second) {
            float root2 = (-b + sdisc) + (do_refine ? root1 : 0);
            if (rtPotentialIntersection(root2)) {
                shading_normal = geometric_normal = (O + root2*D) / radius;
                float s = (shading_normal.z + 1.0f) * 0.5f;
                float t = (atan2(shading_normal.y, shading_normal.x) + PIf) * 0.5f / PIf;
                texcoord = make_float2(s, t);
                rtReportIntersection(0);
            }
        }
    }
}

RT_PROGRAM void intersect(int primIdx) {
    intersect_sphere<false>();
}

RT_PROGRAM void robust_intersect(int primIdx) {
    intersect_sphere<true>();
}

RT_PROGRAM void bounds(int, float result[6]) {
    const float3 center = make_float3(sphere);
    const float3 radius = make_float3(sphere.w);

    optix::Aabb* aabb = (optix::Aabb*)result;

    aabb->m_min = center - radius;
    aabb->m_max = center + radius;
}
