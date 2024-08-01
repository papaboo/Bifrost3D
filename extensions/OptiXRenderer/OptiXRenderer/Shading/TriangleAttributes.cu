// Triangle attribute interpolation program.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <OptiXRenderer/Types.h>

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace OptiXRenderer;
using namespace optix;

rtDeclareVariable(int, mesh_flags, , );
rtBuffer<uint3> index_buffer;
rtBuffer<VertexGeometry> geometry_buffer;
rtBuffer<float2> texcoord_buffer;

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(float2, texcoord, attribute texcoord, );
rtDeclareVariable(unsigned int, primitive_index, attribute primitive_index, );

//-------------------------------------------------------------------------------------------------
// Attribute program for triangle meshes.
//-------------------------------------------------------------------------------------------------
RT_PROGRAM void interpolate_attributes() {
    primitive_index = rtGetPrimitiveIndex();

    const uint3 vertex_indices = index_buffer[primitive_index];
    const VertexGeometry g0 = geometry_buffer[vertex_indices.x];
    const VertexGeometry g1 = geometry_buffer[vertex_indices.y];
    const VertexGeometry g2 = geometry_buffer[vertex_indices.z];
    
    geometric_normal = normalize(cross(g1.position - g0.position, g2.position - g0.position));

    const float2 barycentrics = rtGetTriangleBarycentrics();
    float barycentrics_z = 1.0f - barycentrics.x - barycentrics.y;

    if (mesh_flags & MeshFlags::Normals) {
        shading_normal = g1.normal.decode() * barycentrics.x + g2.normal.decode() * barycentrics.y +
                         g0.normal.decode() * barycentrics_z;
        shading_normal = normalize(shading_normal);
    } else
        shading_normal = geometric_normal;

    if (mesh_flags & MeshFlags::Texcoords) {
        const float2 texcoord0 = texcoord_buffer[vertex_indices.x];
        const float2 texcoord1 = texcoord_buffer[vertex_indices.y];
        const float2 texcoord2 = texcoord_buffer[vertex_indices.z];
        texcoord = texcoord1 * barycentrics.x + texcoord2 * barycentrics.y + texcoord0 * barycentrics_z;
    }
    else
        texcoord - make_float2(0.0f);
}
