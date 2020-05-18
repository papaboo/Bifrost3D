// Sphere light.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include "LightSources.hlsl"

// ------------------------------------------------------------------------------------------------
// Constant buffers.
// ------------------------------------------------------------------------------------------------

cbuffer lights : register(b12) {
    int4 light_count;
    LightData light_data[12];
}

cbuffer scene_variables : register(b13) {
    SceneVariables scene_vars;
};

// ------------------------------------------------------------------------------------------------
// Utils.
// ------------------------------------------------------------------------------------------------

float compute_pixel_depth(float4 world_position, float2 texcoord) {
    float3 to_camera = normalize(scene_vars.camera_position.xyz - world_position.xyz);
    float sphere_radius = world_position.w;
    float offset = sqrt(1.0f - dot(texcoord, texcoord)) * sphere_radius;
    world_position.xyz += to_camera * offset;
    world_position.w = 1.0f;
    float4 projected_position = mul(world_position, scene_vars.view_projection_matrix);
    return projected_position.z / projected_position.w;
}

// ------------------------------------------------------------------------------------------------
// Vertex shader.
// ------------------------------------------------------------------------------------------------

struct Varyings {
    float4 position : SV_POSITION;
    float4 world_position : WORLD_POSITION;
    float2 texcoord : TEXCOORD;
    nointerpolation float3 radiance : COLOR;
};

Varyings vs(uint primitive_ID : SV_VertexID) {
    // Determine which light to render and bail out if it is not a sphere light.
    uint light_index = primitive_ID / 6u;
    LightData light = light_data[light_index];
    if (light.type() != LightType::Sphere)
        return (Varyings)0;

    // Draw quad with two triangles: 
    //   {-1, -1}, {-1, 1}, {1, -1}
    //   {-1, 1}, {1, -1}, {1, 1}
    uint vertex_ID = primitive_ID % 6u;
    bool is_second_triangle = vertex_ID > 2;
    vertex_ID = is_second_triangle ? 6 - vertex_ID : vertex_ID; // Remap from [3, 4, 5] to [3, 2, 1] if it's the second triangle.
    Varyings output;
    output.texcoord.x = vertex_ID < 2 ? -1 : 1;
    output.texcoord.y = (vertex_ID % 2 == 0) ? -1 : 1;

    // Compute position in world space and offset the vertices by the sphere radius along the tangent axis.
    output.world_position.xyz = light.sphere_position();
    float3 forward = normalize(output.world_position.xyz - scene_vars.camera_position.xyz);
    float3 camera_right = scene_vars.world_to_view_matrix._m00_m10_m20;
    float3 light_up = normalize(cross(forward, camera_right));
    float3 light_right = normalize(cross(light_up, forward));
    output.world_position.xyz += (output.texcoord.x * light_right + output.texcoord.y * light_up) * light.sphere_radius();

    output.world_position.w = 1.0f;
    output.position = mul(output.world_position, scene_vars.view_projection_matrix);
    output.world_position.w = light.sphere_radius();

    output.radiance = evaluate_sphere_light(light, scene_vars.camera_position.xyz);

    return output;
}

// ------------------------------------------------------------------------------------------------
// Color shader.
// ------------------------------------------------------------------------------------------------

struct Pixel {
    float4 color : SV_TARGET;
    float depth : SV_DEPTH; // TODO SV_DepthGreater, or SV_DepthLessEqual - source: https://mynameismjp.wordpress.com/2010/11/14/d3d11-features/
};

Pixel color_PS(Varyings input) {
    if (dot(input.texcoord, input.texcoord) > 1.0f)
        discard;

    Pixel pixel;
    pixel.color = float4(input.radiance, 1.0f);
    pixel.depth = compute_pixel_depth(input.world_position, input.texcoord);
    return pixel;
}

// ------------------------------------------------------------------------------------------------
// GBuffer shader.
// ------------------------------------------------------------------------------------------------

struct GBufferPixel {
    float2 normal : SV_TARGET;
    float depth : SV_DEPTH;
};

GBufferPixel g_buffer_PS(Varyings input) {
    float distSqrd = dot(input.texcoord, input.texcoord);
    if (distSqrd > 1.0f)
        discard;

    float3 view_space_normal = float3(input.texcoord, sqrt(1 - distSqrd));

    GBufferPixel pixel;
    pixel.normal.xy = encode_ss_octahedral_normal(view_space_normal);
    pixel.depth = compute_pixel_depth(input.world_position, input.texcoord);
    return pixel;
}