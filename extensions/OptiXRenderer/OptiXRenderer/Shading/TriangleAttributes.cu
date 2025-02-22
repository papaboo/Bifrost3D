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
rtBuffer<uchar4> tint_and_roughness_buffer;

rtDeclareVariable(float3, intersection_point, attribute intersection_point, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(float2, texcoord, attribute texcoord, );
rtDeclareVariable(float4, tint_and_roughness_scale, attribute tint_and_roughness_scale, );
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

    intersection_point = g1.position * barycentrics.x + g2.position * barycentrics.y + g0.position * barycentrics_z;

    if (mesh_flags & MeshFlags::Normals) {
        shading_normal = g1.normal.decode() * barycentrics.x + g2.normal.decode() * barycentrics.y +
                         g0.normal.decode() * barycentrics_z;
        shading_normal = normalize(shading_normal);
    } else
        shading_normal = geometric_normal;

    if (mesh_flags & MeshFlags::Texcoords) {
        float2 texcoord0 = texcoord_buffer[vertex_indices.x];
        float2 texcoord1 = texcoord_buffer[vertex_indices.y];
        float2 texcoord2 = texcoord_buffer[vertex_indices.z];
        texcoord = texcoord1 * barycentrics.x + texcoord2 * barycentrics.y + texcoord0 * barycentrics_z;
    }
    else
        texcoord - make_float2(0.0f);

    if (mesh_flags & MeshFlags::Tints) {
        const float byte_to_float_normalizer = 1.0f / 255.0f;
        uchar4 tint0 = tint_and_roughness_buffer[vertex_indices.x];
        uchar4 tint1 = tint_and_roughness_buffer[vertex_indices.y];
        uchar4 tint2 = tint_and_roughness_buffer[vertex_indices.z];
        tint_and_roughness_scale.x = (tint1.x * barycentrics.x + tint2.x * barycentrics.y + tint0.x * barycentrics_z) * byte_to_float_normalizer;
        tint_and_roughness_scale.y = (tint1.y * barycentrics.x + tint2.y * barycentrics.y + tint0.y * barycentrics_z) * byte_to_float_normalizer;
        tint_and_roughness_scale.z = (tint1.z * barycentrics.x + tint2.z * barycentrics.y + tint0.z * barycentrics_z) * byte_to_float_normalizer;
        tint_and_roughness_scale.w = (tint1.w * barycentrics.x + tint2.w * barycentrics.y + tint0.w * barycentrics_z) * byte_to_float_normalizer;
    } else
        tint_and_roughness_scale = make_float4(1.0f); // Multiplicative identity
}
