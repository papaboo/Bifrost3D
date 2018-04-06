// Sphere light.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2017, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include "LightSources.hlsl"

// ------------------------------------------------------------------------------------------------
// Vertex shader.
// ------------------------------------------------------------------------------------------------

cbuffer scene_variables : register(b0) {
    SceneVariables scene_vars;
};

cbuffer lights : register(b1) {
    int4 light_count;
    LightData light_data[12];
}

// ------------------------------------------------------------------------------------------------
// Vertex shader.
// ------------------------------------------------------------------------------------------------

struct Varyings {
    float4 position : SV_POSITION;
    float4 world_position : WORLD_POSITION;
    float2 texcoord : TEXCOORD;
    float3 radiance : COLOR;
};

Varyings vs(uint primitive_ID : SV_VertexID) {
    // Determine which light to render and bail out if it is not a sphere light.
    uint light_index = primitive_ID / 6u;
    LightData light = light_data[light_index];
    if (light.type() != LightType::Sphere) {
        Varyings output;
        output.position = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return output;
    }

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
    float3x3 tangent_space = create_TBN(normalize(output.world_position.xyz - scene_vars.camera_position.xyz));
    output.world_position.xyz += (output.texcoord.x * tangent_space[0] + output.texcoord.y * tangent_space[1]) * light.sphere_radius();
    output.world_position.w = 1.0f;
    output.position = mul(output.world_position, scene_vars.view_projection_matrix);
    output.world_position.w = light.sphere_radius();

    output.radiance = evaluate_sphere_light(light, scene_vars.camera_position.xyz);

    return output;
}

// ------------------------------------------------------------------------------------------------
// Pixel shader.
// ------------------------------------------------------------------------------------------------

struct Pixel {
    float4 color : SV_TARGET;
    float depth : SV_DEPTH;
};

Pixel ps(Varyings input){
    if (dot(input.texcoord, input.texcoord) > 1.0f)
        discard;

    Pixel pixel;
    float3 to_camera = normalize(scene_vars.camera_position.xyz - input.world_position.xyz);
    float sphere_radius = input.world_position.w;
    float offset = sqrt(1.0f - dot(input.texcoord, input.texcoord)) * sphere_radius;
    input.world_position.xyz += to_camera * offset;
    input.world_position.w = 1.0f;
    float4 projected_position = mul(input.world_position, scene_vars.view_projection_matrix);
    pixel.depth = projected_position.z / projected_position.w;

    pixel.color = float4(input.radiance, 1.0f);
    return pixel;
}