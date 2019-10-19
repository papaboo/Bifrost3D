// DirectX 11 renderer host types.
//-------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
//-------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_TYPES_H_
#define _DX11RENDERER_RENDERER_TYPES_H_

#include <Bifrost/Math/half.h>

#include <DX11Renderer/OwnedResourcePtr.h>

//-------------------------------------------------------------------------------------------------
// Forward declarations.
//-------------------------------------------------------------------------------------------------
struct ID3D11BlendState;
struct ID3D10Blob;
using ID3DBlob = ID3D10Blob;
struct ID3D11Buffer;
struct ID3D11ComputeShader;
struct ID3D11DepthStencilState;
struct ID3D11DepthStencilView;
struct ID3D11Device1;
struct ID3D11DeviceContext1;
struct ID3D11InputLayout;
struct ID3D11PixelShader;
struct ID3D11RasterizerState;
struct ID3D11RenderTargetView;
struct ID3D11Resource;
struct ID3D11SamplerState;
struct ID3D11ShaderResourceView;
struct ID3D11Texture2D;
struct ID3D11VertexShader;
struct ID3D11UnorderedAccessView;

struct IDXGIAdapter;
struct IDXGIAdapter1;
struct IDXGIDevice;
struct IDXGIFactory1;
struct IDXGIFactory2;

//-------------------------------------------------------------------------------------------------
// DX11 enum none flags.
//-------------------------------------------------------------------------------------------------
#define D3D11_BIND_NONE 0
#define D3D11_CPU_ACCESS_NONE 0
#define D3D11_MAP_FLAG_NONE 0
#define D3D11_RESOURCE_MISC_NONE 0
#define D3D11_USAGE_NONE 0

namespace DX11Renderer {

//-------------------------------------------------------------------------------------------------
// Alias owned resource pointers
//-------------------------------------------------------------------------------------------------

using OBlendState = DX11Renderer::OwnedResourcePtr<ID3D11BlendState>;
using OBlob = DX11Renderer::OwnedResourcePtr<ID3DBlob>;
using OBuffer = DX11Renderer::OwnedResourcePtr<ID3D11Buffer>;
using OComputeShader = DX11Renderer::OwnedResourcePtr<ID3D11ComputeShader>;
using ODepthStencilState = DX11Renderer::OwnedResourcePtr<ID3D11DepthStencilState>;
using ODepthStencilView = DX11Renderer::OwnedResourcePtr<ID3D11DepthStencilView>;
using ODevice1 = DX11Renderer::OwnedResourcePtr<ID3D11Device1>;
using ODeviceContext1 = DX11Renderer::OwnedResourcePtr<ID3D11DeviceContext1>;
using OInputLayout = DX11Renderer::OwnedResourcePtr<ID3D11InputLayout>;
using OPixelShader = DX11Renderer::OwnedResourcePtr<ID3D11PixelShader>;
using ORasterizerState = DX11Renderer::OwnedResourcePtr<ID3D11RasterizerState>;
using ORenderTargetView = DX11Renderer::OwnedResourcePtr<ID3D11RenderTargetView>;
using OResource = DX11Renderer::OwnedResourcePtr<ID3D11Resource>;
using OSamplerState = DX11Renderer::OwnedResourcePtr<ID3D11SamplerState>;
using OShaderResourceView = DX11Renderer::OwnedResourcePtr<ID3D11ShaderResourceView>;
using OTexture2D = DX11Renderer::OwnedResourcePtr<ID3D11Texture2D>;
using OVertexShader = DX11Renderer::OwnedResourcePtr<ID3D11VertexShader>;
using OUnorderedAccessView = DX11Renderer::OwnedResourcePtr<ID3D11UnorderedAccessView>;

using ODXGIAdapter = DX11Renderer::OwnedResourcePtr<IDXGIAdapter>;
using ODXGIAdapter1 = DX11Renderer::OwnedResourcePtr<IDXGIAdapter1>;
using ODXGIDevice = DX11Renderer::OwnedResourcePtr<IDXGIDevice>;
using ODXGIFactory1 = DX11Renderer::OwnedResourcePtr<IDXGIFactory1>;
using ODXGIFactory2 = DX11Renderer::OwnedResourcePtr<IDXGIFactory2>;

//-------------------------------------------------------------------------------------------------
// Storage.
//-------------------------------------------------------------------------------------------------

#define CONSTANT_ELEMENT_SIZE 16
#define CONSTANT_BUFFER_ALIGNMENT 256

struct half2 {
    half_float::half x, y;
};

struct int2 {
    int x, y;
};

struct int4 {
    int x, y, z, w;
};

struct float2 {
    float x, y;
};

struct float3 {
    float x, y, z;
};

struct float4 {
    float x, y, z, w;
};

struct AABB {
    float3 min, max;
};

enum class SsaoFilter { Cross, Box };

struct SsaoSettings {
    float world_radius = 0.5f;
    float bias = 0.03f;
    float intensity_scale = 0.125f;
    float falloff = 2.0f;
    unsigned int sample_count = 8u;
    float depth_filtering_percentage = 0.25f;
    SsaoFilter filter_type = SsaoFilter::Box;
    int filter_support = 8;
    float normal_std_dev = 0.1f;
    float plane_std_dev = 0.05f;
};

//-------------------------------------------------------------------------------------------------
// Format conversion.
// See https://github.com/apitrace/dxsdk/blob/master/Include/d3dx_dxgiformatconvert.inl
//-------------------------------------------------------------------------------------------------

struct R11G11B10_Float {
    unsigned int raw;

    R11G11B10_Float() = default;

    R11G11B10_Float(float r, float g, float b) {
        // Pack RGB into R11G11B10. All three channels have a 5 bit exponent and no sign bit.
        // This fits perfectly with the half format, that also has a 5 bit exponent.
        // The solution therefore is to convert the float to half and then using bitmasks 
        // to extract the 5 bit exponent and the first 5/6 bits of the mantissa.
        // Masking, however, yields a result that always truncates the values and is therefore biased towards zero.
        // To avoid this bias, we isolate the ignored decimals and add them once more to the color.
        // Any decimals above 0.5 will then cause the next bit to flip and thus round upwards.
        // If speed is a concern it might be faster to use the conversion tables in half_float directly instead of performing math.
        // See Jeroen van der Zijp's Fast Half Float Conversions, http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
        // The memory layout is confusingly [B10, G11, R11]. Go figure.

        half_float::detail::uint16 biased_blue_raw = half(b).raw() & 0x7FE0;
        half biased_blue;
        memcpy(&biased_blue, &biased_blue_raw, sizeof(half));
        unsigned int blue = (half_float::half(2 * b - biased_blue).raw() & 0x7FE0) << 17;

        half_float::detail::uint16 biased_green_raw = half(g).raw() & 0x7FF0;
        half biased_green;
        memcpy(&biased_green, &biased_green_raw, sizeof(half));
        unsigned int green = (half_float::half(2 * g - biased_green).raw() & 0x7FF0) << 7;

        half_float::detail::uint16 biased_red_raw = half(r).raw() & 0x7FF0;
        half biased_red;
        memcpy(&biased_red, &biased_red_raw, sizeof(half));
        unsigned int red = (half_float::half(2 * r - biased_red).raw() & 0x7FF0) >> 4;

        raw = red | green | blue;
    }
};

struct R10G10B10A2_Unorm {
    unsigned int raw;

    R10G10B10A2_Unorm() = default;

    R10G10B10A2_Unorm(float r, float g, float b, float a = 1.0f) {
        static auto saturate = [](float v) -> float { return fmaxf(0.0f, fminf(v, 1.0f)); };
        static auto to_10bit = [](float v) -> unsigned int { return int(saturate(v) * 1023 + 0.5f); };
        static auto to_4bit = [](float v) -> unsigned int { return int(saturate(v) * 3 + 0.5f); };
        raw = to_10bit(r) | to_10bit(g) << 10 | to_10bit(b) << 20 | to_4bit(a) << 30;
    }
};

//-------------------------------------------------------------------------------------------------
// Model structs.
//-------------------------------------------------------------------------------------------------

struct TextureBound {
    static const unsigned int None = 0;
    static const unsigned int Tint = 1 << 0;
    static const unsigned int Roughness = 1 << 1;
    static const unsigned int Coverage = 1 << 2;
    static const unsigned int Metallic = 1 << 3;
};

struct Dx11Material {
    float3 tint;
    unsigned int textures_bound;
    float roughness;
    float specularity;
    float metallic;
    float coverage;
    float coat;
    float coat_roughness;
    float2 __padding;
};

struct Dx11MaterialTextures {
    unsigned int tint_roughness_index;
    unsigned int coverage_index;
    unsigned int metallic_index;
};

struct Dx11Image {
    OShaderResourceView srv;
};

struct Dx11Texture {
    Dx11Image* image;
    OSamplerState sampler;
};

struct Dx11VertexGeometry {
    float3 position;
    int encoded_normal;
};

struct Dx11Mesh {
    unsigned int index_count;
    unsigned int vertex_count;
    ID3D11Buffer* indices;

    ID3D11Buffer* buffers[2]; // [geometry, texcoords]
    int buffer_count;

    AABB bounds;

    ID3D11Buffer* geometry() { return buffers[0]; }
    ID3D11Buffer** geometry_address() { return buffers + 0; }
    ID3D11Buffer* texcoords() { return buffers[1]; }
    ID3D11Buffer** texcoords_address() { return buffers + 1; }
};

struct Dx11Model {
    struct Properties {
        static const unsigned int None = 0u;
        static const unsigned int Cutout = 1u << 0u;
        static const unsigned int Transparent = 1u << 1u;
        static const unsigned int Destroyed = 1u << 31u;
    };

    unsigned int model_ID;
    unsigned int mesh_ID;
    unsigned int transform_ID;
    unsigned int material_ID;
    unsigned int properties; // NOTE If I really really really wanted to keep this 16 byte aligned (which is nice), then I could store properties in upper 8 bits of the IDs.

    bool is_opaque() { return (properties & (Properties::Cutout | Properties::Transparent)) == 0; }
    bool is_cutout() { return (properties & Properties::Cutout) == Properties::Cutout; }
    bool is_transparent() { return (properties & Properties::Transparent) == Properties::Transparent; }
    bool is_destroyed() { return (properties & Properties::Destroyed) == Properties::Destroyed; }
};

//-------------------------------------------------------------------------------------------------
// Light source structs.
//-------------------------------------------------------------------------------------------------

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