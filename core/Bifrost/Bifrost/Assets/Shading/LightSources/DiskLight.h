// Bifrost disk light for shading.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_LIGHTSOURCES_DISK_LIGHT_H_
#define _BIFROST_ASSETS_SHADING_LIGHTSOURCES_DISK_LIGHT_H_

#include <Bifrost/Assets/Shading/LightSources/SphereLight.h>
#include <Bifrost/Assets/Shading/Utils.h>
#include <Bifrost/Math/Distributions.h>
#include <Bifrost/Math/Disk.h>

#ifndef GPU_COMPILATION
#include <sstream>
#endif

namespace Bifrost::Assets::Shading::LightSources {

struct DiskLight {
private:
    Math::Disk m_surface;
    Math::RGB m_power;
    float m_surface_area;
    bool m_is_two_sided;

public:
    DiskLight() = default;
    DiskLight(Math::Disk surface, Math::RGB power, bool is_two_sided = false)
        : m_surface(surface), m_power(power), m_surface_area(surface.get_surface_area()), m_is_two_sided(is_two_sided) {}
    DiskLight(Math::Vector3f position, Math::Vector3f normal, Math::RGB power, bool is_two_sided = false)
        : m_surface(Math::Disk(position, 0, normal)), m_power(power), m_surface_area(0.0f), m_is_two_sided(is_two_sided) {}

    static DiskLight approximate_sphere_light(SphereLight sphere_light, Math::Vector3f lit_position) {
        auto disk_light_power = sphere_light.get_power() * 0.5f; // Adjust power to account for the twosided disk having half the surface area of the sphere.
        auto surface_normal = normalize(lit_position - sphere_light.get_position());
        auto surface = Math::Disk(sphere_light.get_position(), sphere_light.get_radius(), surface_normal);
        return DiskLight(surface, disk_light_power, true);
    }

    _inline_all_archs_ bool is_delta_light() const {
        // True if the disk has no surface area.
        return m_surface_area == 0;
    }

    _inline_all_archs_ Math::Disk get_surface() const { return m_surface; }
    _inline_all_archs_ Math::Vector3f get_normal() const { return m_surface.normal; }
    _inline_all_archs_ bool is_two_sided() const { return m_is_two_sided; }

    _inline_all_archs_ Math::RGB get_power() const { return m_power; }
    _inline_all_archs_ Math::RGB get_emitted_radiance() const {
        // Convert power of area light to emitted radiance
        // See PBRT v4 end of chapter 12.4. Power = Pi * (twoSided ? 2 : 1) * area * emitted_radiance
        Math::RGB emitted_radiance = m_power / (PIf * m_surface_area);
        return emitted_radiance * (m_is_two_sided ? 0.5f : 1.0f);
    }

    _inline_all_archs_ LightSample sample_radiance(Math::Vector3f lit_position, Math::Vector2f random_sample) const {
        using namespace Bifrost::Math;

        // Check if the point is behind the light source.
        Vector3f direction_to_v0 = m_surface.center - lit_position;
        float signed_distance_to_plane = dot(direction_to_v0, m_surface.normal);
        bool lit_position_in_plane = signed_distance_to_plane == 0.0f;
        bool lit_position_behind = signed_distance_to_plane >= 0.0f;
        if (lit_position_in_plane || (!m_is_two_sided && lit_position_behind))
            return LightSample::none();

        LightSample light_sample;
        if (is_delta_light()) {
            light_sample.direction_to_light = direction_to_v0;
            light_sample.distance = magnitude(light_sample.direction_to_light);
            light_sample.direction_to_light /= light_sample.distance;
            float abs_cos_theta_light = abs(dot(light_sample.direction_to_light, m_surface.normal));
            light_sample.radiance = m_power * abs_cos_theta_light / (PIf * pow2(light_sample.distance));
            light_sample.PDF = MonteCarlo::PDF::delta_dirac(1);
        } else {
            // Sample the triangle.
            auto triangle_sample = Distributions::Disk::sample_solid_angle(lit_position, m_surface.center, m_surface.radius, m_surface.normal, random_sample);

            light_sample.distance = magnitude(triangle_sample.direction);
            light_sample.direction_to_light = triangle_sample.direction / light_sample.distance;
            light_sample.PDF = triangle_sample.PDF;

            if (isinf(light_sample.PDF.value()))
                // Lit position is in the plane spanned by the triangle.
                return LightSample::none();

            // Compute radiance.
            Math::RGB emitted_power = m_power / (PIf * m_surface_area);
            light_sample.radiance = emitted_power;
        }

        // Account for light being distributed across both sides.
        light_sample.radiance *= m_is_two_sided ? 0.5f : 1.0f;

        // Decrement the shadow ray distance by one ULP. This should avoid self intersections,
        // as it's the same intersection test that is used in the intersection program.
        light_sample.distance = nextafterf(light_sample.distance, 0.0f);

        return light_sample;
    }

    _inline_all_archs_ Math::MonteCarlo::PDF pdf(Math::Vector3f lit_position, Math::Vector3f direction_to_light) const {
        using namespace Bifrost::Math;

        // Check if the point is behind the light source.
        bool lit_position_behind = dot(direction_to_light, m_surface.normal) >= 0.0f;
        if (lit_position_behind && !m_is_two_sided)
            return Math::MonteCarlo::PDF::invalid();

        if (is_delta_light())
            return Math::MonteCarlo::PDF::delta_dirac(0);
        else
            return Distributions::Disk::solid_angle_PDF(lit_position, direction_to_light, m_surface.center, m_surface.radius, m_surface.normal);
    }

    _inline_all_archs_ LightResponse evaluate_with_PDF(Math::Vector3f lit_position, Math::Vector3f direction_to_light) const {
        auto PDF = pdf(lit_position, direction_to_light);
        if (PDF.invalid_or_delta_dirac())
            return { Math::RGB::black(), PDF };

        return { get_emitted_radiance(), PDF };
    }

#ifndef GPU_COMPILATION
    inline std::string to_string() const {
        std::ostringstream out;
        out << "[DiskLight: surface: " << m_surface.to_string() << ", power: " << m_power << "]";
        return out.str();
    }
#endif
};

} // NS Bifrost::Assets::Shading::LightSources

#endif // _BIFROST_ASSETS_SHADING_LIGHTSOURCES_TRIANGLE_LIGHT_H_