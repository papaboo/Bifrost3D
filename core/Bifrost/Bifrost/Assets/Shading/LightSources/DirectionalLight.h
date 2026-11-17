// Bifrost directional light for shading.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_LIGHTSOURCES_DIRECTIONAL_LIGHT_H_
#define _BIFROST_ASSETS_SHADING_LIGHTSOURCES_DIRECTIONAL_LIGHT_H_

#include <Bifrost/Math/Color.h>
#include <Bifrost/Assets/Shading/Utils.h>

#ifndef GPU_COMPILATION
#include <sstream>
#endif

namespace Bifrost::Assets::Shading::LightSources {

struct DirectionalLight {
private:
    Math::Vector3f m_direction;
    Math::RGB m_radiance;

public:
    DirectionalLight() = default;
    DirectionalLight(Math::Vector3f direction, Math::RGB radiance)
        : m_direction(direction), m_radiance(radiance) { }


    _inline_all_archs_ Math::Vector3f get_direction() const { return m_direction; }
    _inline_all_archs_ Math::RGB get_emitted_radiance() const { return m_radiance; }
    _inline_all_archs_ bool is_delta_light() const { return true; } // A directional light is always a delta light as it has no surface area or light distribution.

    _inline_all_archs_ LightSample sample_radiance() const {
        LightSample sample;
        sample.radiance = m_radiance;
        sample.PDF = Math::MonteCarlo::PDF::delta_dirac(1.0f);
        sample.direction_to_light = -m_direction;
        sample.distance = INFINITY;
        return sample;
    }

    // Overload that follows the general light interface.
    _inline_all_archs_ LightSample sample_radiance(Math::Vector3f lit_position, Math::Vector2f random_sample) const { return sample_radiance(); }

    _inline_all_archs_ Math::MonteCarlo::PDF pdf(Math::Vector3f lit_position, Math::Vector3f direction_to_light) const {
        return Math::MonteCarlo::PDF::delta_dirac(0.0f);
    }

    _inline_all_archs_ LightResponse evaluate_with_PDF(Math::Vector3f lit_position, Math::Vector3f direction_to_light) const {
        return LightResponse::delta_dirac();
    }

#ifndef GPU_COMPILATION
    inline std::string to_string() const {
        std::ostringstream out;
        out << "[DirectionalLight: direction: " << m_direction << ", radiance: " << m_radiance << "]";
        return out.str();
    }
#endif
};

} // NS Bifrost::Assets::Shading::LightSources

#endif // _BIFROST_ASSETS_SHADING_LIGHTSOURCES_DIRECTIONAL_LIGHT_H_