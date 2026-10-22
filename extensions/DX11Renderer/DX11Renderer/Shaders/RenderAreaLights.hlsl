// Render area lights.
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

float compute_sphere_pixel_depth(float3 world_position, float radius, float2 texcoord) {
    float3 to_camera = normalize(scene_vars.camera_world_position() - world_position);
    float offset = sqrt(1.0f - dot(texcoord, texcoord)) * radius;
    world_position += to_camera * offset;
    float4 projected_position = mul(float4(world_position, 1), scene_vars.view_projection_matrix);
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
    // Determine which light to render and bail out if it is not a spherical area light.
    uint light_index = primitive_ID / 6u;
    LightData general_light = light_data[light_index];
    if (!general_light.is_sphere_light() && !general_light.is_spot_light())
        return (Varyings)0;

    float radius = general_light.is_spot_light() ? general_light.spot_radius() : general_light.sphere_radius();
    if (radius == 0)
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
    float3 center, forward;
    if (general_light.is_spot_light()) {
        center = general_light.spot_position();
        forward = general_light.spot_direction();

        // Ensure the disk turns towards the camera
        float3 camera_forward = center - scene_vars.camera_world_position();
        if (dot(forward, camera_forward) < 0)
            forward = -forward;
    } else {
        // Sphere light
        center = general_light.sphere_position();
        forward = normalize(center - scene_vars.camera_world_position());
    }
    
    float3 camera_right = scene_vars.world_to_view_matrix._m00_m10_m20;
    float3 light_up = normalize(cross(forward, camera_right));
    float3 light_right = normalize(cross(light_up, forward));
    output.world_position.xyz = center + (output.texcoord.x * light_right + output.texcoord.y * light_up) * radius;

    output.world_position.w = 1.0f;
    output.position = mul(output.world_position, scene_vars.view_projection_matrix);
    // Set radius to zero for disk lights, as that leaves the depth as-is in the pixel shader.
    output.world_position.w = general_light.is_spot_light() ? 0.0 : radius;

    output.radiance = general_light.is_spot_light() ? general_light.spot_emitted_radiance() : general_light.sphere_emitted_radiance();

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
    pixel.depth = compute_sphere_pixel_depth(input.world_position.xyz, input.world_position.w, input.texcoord);
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
    pixel.depth = compute_sphere_pixel_depth(input.world_position.xyz, input.world_position.w, input.texcoord);
    return pixel;
}