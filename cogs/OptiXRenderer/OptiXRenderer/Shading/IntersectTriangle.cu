// OptiX triangle intersection program.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------
// Inspired by the OptiX samples.
// ---------------------------------------------------------------------------

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

using namespace optix;

rtDeclareVariable(Ray, ray, rtCurrentRay, );

rtBuffer<int3>   index_buffer;
rtBuffer<float3> vertex_buffer;
rtBuffer<float3> normal_buffer;
rtBuffer<float2> texcoord_buffer;

rtDeclareVariable(float2, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 

RT_PROGRAM void mesh_intersect(int primitiveIndex) {
    const int3 vertex_index = index_buffer[primitiveIndex];

    const float3 p0 = vertex_buffer[vertex_index.x];
    const float3 p1 = vertex_buffer[vertex_index.y];
    const float3 p2 = vertex_buffer[vertex_index.z];

    // Intersect ray with triangle
    float3 geo_normal;
    float t, beta, gamma;
    if (intersect_triangle(ray, p0, p1, p2, geo_normal, t, beta, gamma)) {
        if (rtPotentialIntersection(t)) {

            geometric_normal = normalize(geo_normal);

            const float3 n0 = normal_buffer[vertex_index.x];
            const float3 n1 = normal_buffer[vertex_index.y];
            const float3 n2 = normal_buffer[vertex_index.z];
            shading_normal = normalize(n1*beta + n2*gamma + n0*(1.0f - beta - gamma));

            const float2 t0 = texcoord_buffer[vertex_index.x];
            const float2 t1 = texcoord_buffer[vertex_index.y];
            const float2 t2 = texcoord_buffer[vertex_index.z];
            texcoord = t1*beta + t2*gamma + t0*(1.0f - beta - gamma);

            rtReportIntersection(0);
        }
    }
}

RT_PROGRAM void mesh_bounds(int primIdx, float result[6]) {
    const int3 vertex_index = index_buffer[primIdx];

    const float3 v0 = vertex_buffer[vertex_index.x];
    const float3 v1 = vertex_buffer[vertex_index.y];
    const float3 v2 = vertex_buffer[vertex_index.z];
    const float area = length(cross(v1 - v0, v2 - v0));

    optix::Aabb* aabb = (optix::Aabb*)result;

    if (area > 0.0f && !isinf(area)) {
        aabb->m_min = fminf(fminf(v0, v1), v2);
        aabb->m_max = fmaxf(fmaxf(v0, v1), v2);
    } else {
        aabb->invalidate();
    }
}
