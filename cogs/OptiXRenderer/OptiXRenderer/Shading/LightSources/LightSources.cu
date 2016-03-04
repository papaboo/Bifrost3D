// OptiX light source intersection programs.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Shading/LightSources/PointLightImpl.h>

#include <optix.h>
#include <optixu/optixu_aabb.h>
#include <optixu/optixu_math.h>

using namespace OptiXRenderer;
using namespace optix;

rtDeclareVariable(Ray, ray, rtCurrentRay, );

rtBuffer<PointLight, 1> g_lights;

// Encode light index in geometric_normal.x
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );

//=============================================================================
// Point light intersection programs.
//=============================================================================
RT_PROGRAM void intersect(int primitive_index) {

    const PointLight& light = g_lights[primitive_index];

    // TODO Move sphere intersection into util function and use here and in IntersectSphere.cu.
    float3 O = ray.origin - light.position;
    float3 D = ray.direction;

    float b = dot(O, D);
    float c = dot(O, O) - light.radius * light.radius;
    float disc = b * b - c;
    if (disc > 0.0f) {
        float sdisc = sqrtf(disc);
        float root1 = (-b - sdisc);
        float root11 = 0.0f;
        bool check_second = true;
        if (rtPotentialIntersection(root1 + root11)) {
            geometric_normal.x = __int_as_float(primitive_index);
            if (rtReportIntersection(0))
                check_second = false;
        }
        if (check_second) {
            float root2 = (-b + sdisc);
            if (rtPotentialIntersection(root2)) {
                geometric_normal.x = __int_as_float(primitive_index);
                rtReportIntersection(0);
            }
        }
    }
}

RT_PROGRAM void bounds(int primitive_index, float result[6]) {
    const PointLight& light = g_lights[primitive_index];

    optix::Aabb* aabb = (optix::Aabb*)result;

    if (light.radius > 0.0f) {
        aabb->m_min = light.position - light.radius;
        aabb->m_max = light.position + light.radius;
    } else
        aabb->invalidate();
}