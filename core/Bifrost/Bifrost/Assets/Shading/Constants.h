// Bifrost shading utilities
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_CONSTANTS_H_
#define _BIFROST_ASSETS_SHADING_CONSTANTS_H_

#include <Bifrost/Assets/Shading/Utils.h>

namespace Bifrost::Assets::Shading {

// ---------------------------------------------------------------------------
// Indices of refraction
// ---------------------------------------------------------------------------
constexpr float air_ior = 1.0003f;
constexpr float ice_ior = 1.31f;
constexpr float water_ior = 1.33f;
constexpr float coat_ior = 1.5f;
constexpr float glass_ior = 1.52f;
constexpr float diamond_ior = 2.42f;

// ---------------------------------------------------------------------------
// Specularities of materials in air
// ---------------------------------------------------------------------------
constexpr float default_specularity = 0.04f;
constexpr float coat_specularity = 0.04f;
constexpr float ice_specularity = dielectric_specularity(air_ior, ice_ior);
constexpr float water_specularity = dielectric_specularity(air_ior, water_ior);
constexpr float glass_specularity = dielectric_specularity(air_ior, glass_ior);
constexpr float diamond_specularity = dielectric_specularity(air_ior, diamond_ior);

// ---------------------------------------------------------------------------
// Metal tints.
// Source: https://dev.epicgames.com/documentation/en-us/unreal-engine/physically-based-materials-in-unreal-engine
// ---------------------------------------------------------------------------
constexpr Math::RGB iron_tint = Math::RGB(0.560f, 0.570f, 0.580f);
constexpr Math::RGB silver_tint = Math::RGB(0.972f, 0.960f, 0.915f);
constexpr Math::RGB aluminum_tint = Math::RGB(0.913f, 0.921f, 0.925f);
constexpr Math::RGB gold_tint = Math::RGB(1.000f, 0.766f, 0.336f);
constexpr Math::RGB copper_tint = Math::RGB(0.955f, 0.637f, 0.538f);
constexpr Math::RGB chromium_tint = Math::RGB(0.550f, 0.556f, 0.554f);
constexpr Math::RGB nickel_tint = Math::RGB(0.660f, 0.609f, 0.526f);
constexpr Math::RGB titanium_tint = Math::RGB(0.542f, 0.497f, 0.449f);
constexpr Math::RGB cobalt_tint = Math::RGB(0.662f, 0.655f, 0.634f);
constexpr Math::RGB platinum_tint = Math::RGB(0.672f, 0.637f, 0.585f);

// ---------------------------------------------------------------------------
// Index of refraction and extinction coefficient for metals at wavelengths 630nm (red), 532nm (green) and 465nm (blue)
// ---------------------------------------------------------------------------
constexpr Math::RGB gold_ior = { 0.1986f, 0.54463f, 1.2515f };
constexpr Math::RGB gold_extinction = { 3.228f, 2.1406f, 1.7517f };
constexpr Math::RGB titanium_ior = { 2.6979f, 2.4793f, 2.3050f };
constexpr Math::RGB titanium_extinction = { 3.7571f, 3.3511f, 3.0820f };

} // NS Bifrost::Assets::Shading

#endif // _BIFROST_ASSETS_SHADING_CONSTANTS_H_