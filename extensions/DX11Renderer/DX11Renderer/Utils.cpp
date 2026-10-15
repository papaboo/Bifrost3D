// DirectX 11 renderer utility functions.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <DX11Renderer/Utils.h>

#include <Bifrost/Assets//Shading//LightSources/LtcAreaLight.h>

#include <filesystem>
#include <string>

using namespace std::filesystem;

namespace DX11Renderer {

void CHECK_HRESULT(HRESULT hr, const char* const file, int line) {
    if (FAILED(hr)) {
        std::string error = "[file:" + std::string(file) + " line:" + std::to_string(line) + "] DX 11";
        switch (hr) {
        case E_INVALIDARG:
            error += " invalid arg.";
            break;
        case DXGI_ERROR_INVALID_CALL:
            error += " DXGI error invalid call.";
            break;
        default:
            error += " unknown HRESULT code: " + std::to_string(hr);
            break;
        }
        fprintf(stderr, "%s\n", error.c_str());
        throw std::exception(error.c_str());
    }
}

ODevice1 get_device1(ID3D11DeviceContext1& context) {
    ID3D11Device* basic_device;
    context.GetDevice(&basic_device);
    ODevice1 device1;
    THROW_DX11_ERROR(basic_device->QueryInterface(IID_PPV_ARGS(&device1)));
    basic_device->Release();
    return device1;
}

void get_data_directory(path& data_directory) {
    char application_path[512];
    GetModuleFileName(nullptr, application_path, 512);
    auto application_directory = path(application_path).parent_path();
    data_directory = application_directory.parent_path() / "Data";
}

OShaderResourceView create_diffuse_sphere_cap_integral_srv(ID3D11Device1& device) {
    using namespace Bifrost::Assets::Shading::LightSources;
    using namespace Bifrost::Math;

    const unsigned int width = diffuse_sphere_cap_integral_texture_size;
    const unsigned int height = diffuse_sphere_cap_integral_texture_size;
    const unsigned int element_count = width * height;

    UNorm8* pixels = new UNorm8[element_count];
    for (unsigned int y = 0; y < height; ++y) {
        float form_factor = y / float(height - 1);
        for (unsigned int x = 0; x < width; ++x) {

            float direction_u = x / float(width - 1);
            float average_direction_z = direction_u * 2 - 1;
            pixels[x + y * width] = LtcAreaLight::integrate_diffuse_sphere_cap(average_direction_z, form_factor);
        }
    }

    OShaderResourceView diffuse_sphere_cap_integral_srv;
    create_texture_2D(device, DXGI_FORMAT_R8_UNORM, pixels, width, height, D3D11_USAGE_IMMUTABLE, &diffuse_sphere_cap_integral_srv);

    delete[] pixels;

    return diffuse_sphere_cap_integral_srv;
}

} // NS DX11Renderer
