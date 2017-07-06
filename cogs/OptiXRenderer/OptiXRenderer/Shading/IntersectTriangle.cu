// OptiX triangle intersection program.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------
// Inspired by the OptiX samples.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Types.h>

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

using namespace OptiXRenderer;
using namespace optix;

rtDeclareVariable(Ray, ray, rtCurrentRay, );

rtDeclareVariable(int, mesh_flags, , );
rtBuffer<uint3> index_buffer;
rtBuffer<VertexGeometry> geometry_buffer;
rtBuffer<float2> texcoord_buffer;

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(float2, texcoord, attribute texcoord, );

//----------------------------------------------------------------------------
// Intersection program for triangle meshes.
// Future work:
// * Test if interleaving the vertex attributes will improve performance.
//----------------------------------------------------------------------------
RT_PROGRAM void intersect(int primitive_index) {
    const uint3 vertex_index = index_buffer[primitive_index];

    const VertexGeometry g0 = geometry_buffer[vertex_index.x];
    const VertexGeometry g1 = geometry_buffer[vertex_index.y];
    const VertexGeometry g2 = geometry_buffer[vertex_index.z];

    // Intersect ray with triangle.
    float3 geo_normal;
    float t, beta, gamma;
    if (intersect_triangle(ray, g0.position, g1.position, g2.position, geo_normal, t, beta, gamma)) {
        if (rtPotentialIntersection(t)) {

            geometric_normal = normalize(geo_normal);

            if (mesh_flags & MeshFlags::Normals) {
                const float3 n0 = g0.normal.decode_unnormalized();
                const float3 n1 = g1.normal.decode_unnormalized();
                const float3 n2 = g2.normal.decode_unnormalized();
                shading_normal = normalize(n1*beta + n2*gamma + n0*(1.0f - beta - gamma));
            } else
                shading_normal = geometric_normal;

            if (mesh_flags & MeshFlags::Texcoords) {
                const float2 t0 = texcoord_buffer[vertex_index.x];
                const float2 t1 = texcoord_buffer[vertex_index.y];
                const float2 t2 = texcoord_buffer[vertex_index.z];
                texcoord = t1*beta + t2*gamma + t0*(1.0f - beta - gamma);
            } else
                texcoord = make_float2(0.0f);

            rtReportIntersection(0);
        }
    }
}

RT_PROGRAM void bounds(int primitive_index, float result[6]) {
    const uint3 vertex_index = index_buffer[primitive_index];

    const float3 v0 = geometry_buffer[vertex_index.x].position;
    const float3 v1 = geometry_buffer[vertex_index.y].position;
    const float3 v2 = geometry_buffer[vertex_index.z].position;

    Aabb* aabb = (Aabb*)result;

    float3 up = cross(v1 - v0, v2 - v0);
    float normal_length_squared = dot(up, up);
    if (normal_length_squared > 0.0f && !isinf(normal_length_squared)) {
        aabb->m_min = fminf(fminf(v0, v1), v2);
        aabb->m_max = fmaxf(fmaxf(v0, v1), v2);
    } else
        aabb->invalidate();
}
