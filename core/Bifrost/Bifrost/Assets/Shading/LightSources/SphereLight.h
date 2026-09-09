// Bifrost sphere light for shading.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_LIGHTSOURCES_SPHERE_LIGHT_H_
#define _BIFROST_ASSETS_SHADING_LIGHTSOURCES_SPHERE_LIGHT_H_

#include <Bifrost/Math/Distributions.h>
#include <Bifrost/Math/Intersect.h>
#include <Bifrost/Assets/Shading/Utils.h>

#ifndef GPU_COMPILATION
#include <sstream>
#endif

namespace Bifrost::Assets::Shading::LightSources {

struct SphereLight {
private:
    Math::Vector3f m_position;
    float m_radius;
    Math::RGB m_power;

    constexpr _inline_all_archs_ float sphere_light_small_sin_theta_squared() const { return 0.0f; } // 1e-5f;

public:
    SphereLight() = default;
    SphereLight(Math::Vector3f position, float radius, Math::RGB power)
        : m_position(position), m_radius(radius), m_power(power) { }


    _inline_all_archs_ Math::Vector3f get_position() const { return m_position; }
    _inline_all_archs_ Math::RGB get_power() const { return m_power; }
    _inline_all_archs_ float get_radius() const { return m_radius; }
    _inline_all_archs_ float get_surface_area() const { return 4.0f * PIf * m_radius * m_radius; }

    _inline_all_archs_ Math::RGB get_emitted_radiance() const {
        // Convert power of area light to emitted radiance
        // See PBRT v4 end of chapter 12.4. Power = Pi * (twoSided ? 2 : 1) * area * emitted_radiance
        Math::RGB emitted_radiance = m_power / (PIf * get_surface_area());
        return emitted_radiance;
    }

    // Returns true if the sphere light should be interpreted as a delta light / point light.
    // Ideally this should only happen if the radius is zero, but due to floating point 
    // imprecision when sampling cones, we draw the line at very tiny subtended angles.
    // TODO Rethink this fix next time it's reproduced, as the fix caused the small light 
    //      in the Veach scene to be too dim and not produce bloom.
    //      There are at least 3 'unguarded' sqrt functions when sampling cones, 
    //      where the input has a risc of becoming negative.
    _inline_all_archs_ bool is_delta_light(Math::Vector3f lit_position) const {
        Math::Vector3f vector_to_light = m_position - lit_position;
        float sin_theta_squared = m_radius * m_radius / dot(vector_to_light, vector_to_light);
        return sin_theta_squared <= sphere_light_small_sin_theta_squared();
    }

    _inline_all_archs_ LightSample sample_radiance(Math::Vector3f lit_position, Math::Vector2f random_sample) const {

        // Sample Sphere light by sampling a cone with the angle subtended by the sphere.
        Math::Vector3f vector_to_light = m_position - lit_position;

        float sin_theta_squared = m_radius * m_radius / dot(vector_to_light, vector_to_light);

        LightSample light_sample;
        if (sin_theta_squared <= sphere_light_small_sin_theta_squared()) {
            // If the subtended angle is too small, then sampling produces NaN's, so just fall back to a point light.
            light_sample.direction_to_light = vector_to_light;
            light_sample.distance = magnitude(light_sample.direction_to_light);
            light_sample.direction_to_light /= light_sample.distance;
            // Radiance is expressed in terms of light emitted per surface area, which isn't valid for lights with no surface.
            // Instead we 'fudge it' by computing the radiant intensity of the light over the squared distance,
            // which ensures that sampling a delta light and an infinitely small area light gives the same result.
            // See the PBRT chapter about point lights for reference.
            light_sample.radiance = m_power / (4.0f * PIf * Math::pow2(light_sample.distance));
            light_sample.distance -= m_radius;
            light_sample.PDF = Math::MonteCarlo::PDF::delta_dirac(1);
        } else {
            // Sample the cone and project the sample onto the sphere.
            float cos_theta = sqrtf(1.0f - sin_theta_squared);
            auto cone_sample = Math::Distributions::Cone::sample(cos_theta, random_sample);

            Math::Matrix3x3f tbn = compute_TBN(normalize(vector_to_light));
            light_sample.direction_to_light = cone_sample.direction * tbn;
            light_sample.PDF = cone_sample.PDF;
            light_sample.distance = Math::Intersect::ray_sphere(lit_position, light_sample.direction_to_light, m_position, m_radius);
            if (light_sample.distance <= 0.0f)
                // The ray missed the sphere, but since it was sampled to be inside the sphere, just assume that it hit at a grazing angle.
                light_sample.distance = dot(vector_to_light, light_sample.direction_to_light);

            light_sample.radiance = get_emitted_radiance();
        }

        // Decrement the shadow ray distance by one ULP. This should avoid self intersections,
        // as it's the same intersection test that is used in the intersection program.
        light_sample.distance = nextafterf(light_sample.distance, 0.0f);

        return light_sample;
    }

    _inline_all_archs_ Math::MonteCarlo::PDF pdf(Math::Vector3f lit_position, Math::Vector3f direction_to_light) const {
        Math::Vector3f vector_to_light_center = m_position - lit_position;

        float sin_theta_squared = m_radius * m_radius / dot(vector_to_light_center, vector_to_light_center);
        bool is_delta_light = sin_theta_squared <= sphere_light_small_sin_theta_squared();
        if (is_delta_light)
            return Math::MonteCarlo::PDF::delta_dirac(0);
        else {
            float cos_theta_max = sqrtf(1.0f - sin_theta_squared);
            float cos_theta = dot(direction_to_light, normalize(vector_to_light_center));

            float is_valid = cos_theta >= cos_theta_max ? 1.0f : 0.0f;
            float PDF = Math::Distributions::Cone::PDF(cos_theta_max) * is_valid;

            return PDF;
        }
    }

    _inline_all_archs_ LightResponse evaluate_with_PDF(Math::Vector3f lit_position, Math::Vector3f direction_to_light) const {
        auto PDF = pdf(lit_position, direction_to_light);
        if (PDF.invalid_or_delta_dirac())
            return { Math::RGB::black(), PDF };

        return { get_emitted_radiance(), PDF};
    }

#ifndef GPU_COMPILATION
    inline std::string to_string() const {
        std::ostringstream out;
        out << "[SphereLight: position: " << m_position << ", radius: " << m_radius << ", power: " << m_power << "]";
        return out.str();
    }
#endif
};

} // NS Bifrost::Assets::Shading::LightSources

#endif // _BIFROST_ASSETS_SHADING_LIGHTSOURCES_SPHERE_LIGHT_H_