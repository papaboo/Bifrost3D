// DirectX 11 renderer host types.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_TYPES_H_
#define _DX11RENDERER_RENDERER_TYPES_H_

#include <Cogwheel/Math/Color.h>
#include <Cogwheel/Math/Vector.h>

struct ID3D11Buffer;

namespace DX11Renderer {

//----------------------------------------------------------------------------
// Model structs.
//----------------------------------------------------------------------------

struct Dx11Material {
    Cogwheel::Math::RGB tint;
    unsigned int tint_texture_index;
    float roughness;
    float specularity;
    float metallic;
    float coverage;
    unsigned int coverage_texture_index;
};

struct Dx11Image {
    ID3D11Texture2D* texture2D;
    ID3D11ShaderResourceView* srv;
};

struct Dx11Texture {
    Dx11Image* image;
    ID3D11SamplerState* sampler;
};

struct Dx11Mesh {
    unsigned int index_count;
    unsigned int vertex_count;
    ID3D11Buffer* indices;

    ID3D11Buffer* buffers[3]; // [positions, normals, texcoords]
    unsigned int strides[3];
    unsigned int offsets[3];
    int buffer_count;

    ID3D11Buffer* positions() { return buffers[0]; }
    ID3D11Buffer** positions_address() { return buffers; }
    ID3D11Buffer* normals() { return buffers[1]; }
    ID3D11Buffer** normals_address() { return buffers + 1; }
    ID3D11Buffer* texcoords() { return buffers[2]; }
    ID3D11Buffer** texcoords_address() { return buffers + 2; }
};

struct Dx11Model {
    unsigned int mesh_ID;
    unsigned int transform_ID;
    unsigned int material_ID;
};

//----------------------------------------------------------------------------
// Light source structs.
//----------------------------------------------------------------------------

struct float3 {
    float x;  float y; float z;
};

struct Dx11SphereLight {
    float3 power;
    float3 position;
    float radius;
};

struct Dx11DirectionalLight {
    float3 radiance;
    float3 direction;
    float __padding;
};

struct Dx11Light{
    enum Flags {
        None = 0u,
        Sphere = 1u,
        Directional = 2u,
        TypeMask = 3u
    };

    float flags;

    union {
        Dx11SphereLight sphere;
        Dx11DirectionalLight directional;
    };

    unsigned int get_type() const { return (Flags)(unsigned int)flags; }
    bool is_type(Flags light_type) { return get_type() == light_type; }
};

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_TYPES_H_