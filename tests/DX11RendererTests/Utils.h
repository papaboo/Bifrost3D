// DX11Renderer testing utils.
// -------------------------------------------------------------------------------------------------
// Copyright (C) 2015-2017, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// -------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERERTEST_UTILS_H_
#define _DX11RENDERERTEST_UTILS_H_

#define NOMINMAX
#include <D3D11_1.h>
#undef RGB

#include <DX11Renderer\Types.h>

#include <algorithm>
#include <vector>

// -------------------------------------------------------------------------------------------------
// Headless DX11 context creation
// -------------------------------------------------------------------------------------------------

struct DeviceInitilization1 {
    DX11Renderer::OID3D11Device1 device;
    DX11Renderer::OID3D11DeviceContext1 context;
};

// TODO Move into compositor as helper function.
inline DeviceInitilization1 create_headless_device1() {
    // Find the best performing device (apparently the one with the most memory) and initialize that.
    struct WeightedAdapter {
        int index, dedicated_memory;

        inline bool operator<(WeightedAdapter rhs) const {
            return rhs.dedicated_memory < dedicated_memory;
        }
    };

    IDXGIAdapter1* adapter = nullptr;

    IDXGIFactory1* dxgi_factory1;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgi_factory1));

    std::vector<WeightedAdapter> sorted_adapters;
    for (int adapter_index = 0; dxgi_factory1->EnumAdapters1(adapter_index, &adapter) != DXGI_ERROR_NOT_FOUND; ++adapter_index) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);

        // Ignore software rendering adapters. TODO Just give them a low priority instead of completely ignoring them.
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            continue;

        WeightedAdapter e = { adapter_index, int(desc.DedicatedVideoMemory >> 20) };
        sorted_adapters.push_back(e);
    }

    std::sort(sorted_adapters.begin(), sorted_adapters.end());

    // Then create the device and render context.
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* immediate_context = nullptr;
    for (WeightedAdapter a : sorted_adapters) {
        dxgi_factory1->EnumAdapters1(a.index, &adapter);

        UINT create_device_flags = 0; // D3D11_CREATE_DEVICE_DEBUG;
        D3D_FEATURE_LEVEL feature_level_requested = D3D_FEATURE_LEVEL_11_0;

        D3D_FEATURE_LEVEL feature_level;
        hr = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, create_device_flags, &feature_level_requested, 1,
            D3D11_SDK_VERSION, &device, &feature_level, &immediate_context);

        if (SUCCEEDED(hr))
            break;
    }
    dxgi_factory1->Release();

    DeviceInitilization1 res;
    hr = device->QueryInterface(IID_PPV_ARGS(&res.device));
    // THROW_ON_FAILURE(hr);

    hr = immediate_context->QueryInterface(IID_PPV_ARGS(&res.context));
    // THROW_ON_FAILURE(hr);

    return res;
}

// -------------------------------------------------------------------------------------------------
// Comparison helpers.
// -------------------------------------------------------------------------------------------------

inline bool almost_equal_eps(float lhs, float rhs, float eps) {
    return lhs < rhs + eps && lhs + eps > rhs;
}

#define EXPECT_FLOAT_EQ_EPS(expected, actual, epsilon) EXPECT_PRED3(almost_equal_eps, expected, actual, epsilon)

inline bool almost_equal_percentage(float lhs, float rhs, float percentage) {
    float eps = lhs * percentage;
    return almost_equal_eps(lhs, rhs, eps);
}

#define EXPECT_FLOAT_EQ_PCT(expected, actual, percentage) EXPECT_PRED3(almost_equal_percentage, expected, actual, percentage)

#endif // _DX11RENDERERTEST_UTILS_H_