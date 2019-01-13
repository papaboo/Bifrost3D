// Bifrost transformations.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_MATH_TRANSFORM_H_
#define _BIFROST_MATH_TRANSFORM_H_

#include <Bifrost/Math/Quaternion.h>
#include <Bifrost/Math/Vector.h>

#include <cstring>
#include <sstream>

namespace Bifrost {
namespace Math {

//----------------------------------------------------------------------------
// A 3 dimensional affine transform.
// The rotation is given by a quaternion, the position by a vector3 
// and the transform can be uniformly scaled, 
// which leaves out shearing and non-uniform scaling.
//----------------------------------------------------------------------------
struct Transform final {
    //*****************************************************************************
    // Public members
    //*****************************************************************************
    Quaternionf rotation;
    Vector3f translation;
    float scale;

    // Constructor.
    Transform() = default;
    Transform(Vector3f translation, Quaternionf rotation = Quaternionf::identity(), float scale = 1.0f)
        : rotation(rotation), translation(translation), scale(scale) { }

    static inline Transform identity() {
        return Transform(Vector3f::zero(), Quaternionf::identity(), 1.0f);
    }

    // Computes the delta transform that multiplied onto 'from' produces 'to'.
    // I.e. from * delta = to.
    static inline Transform delta(Transform from, Transform to) {
        return from.inverse() * to;
    }

    //*****************************************************************************
    // Comparison operators.
    //*****************************************************************************
    inline bool operator==(Transform rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) == 0;
    }
    inline bool operator!=(Transform rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) != 0;
    }

    // Apply the transform to a vector.
    inline Vector3f apply(Vector3f v) const {
        return translation + rotation * v * scale;
    }

    // Shorthand overloaded multiplication operator for applying the transform to a vector.
    inline Vector3f operator*(Vector3f rhs) const {
        return apply(rhs);
    }

    // Apply the transform to another transform.
    inline Transform apply(Transform t) const {
        return Transform(this->apply(t.translation), normalize(rotation * t.rotation), scale * t.scale);
    }
    // Shorthand overloaded multiplication operator for applying the transform to another transform.
    inline Transform operator*(Transform rhs) const {
        return apply(rhs);
    }

    // Rotate transform to look at the target.
    inline void look_at(Vector3f target, Vector3f up = Vector3f::up()) {
        Vector3f direction = normalize(target - translation);
        rotation = Quaternionf::look_in(direction, up);
    }

    inline std::string to_string() const {
        std::ostringstream out;
        out << "[translation: " << translation << ", rotation: " << rotation << ", scale: " << scale << "]";
        return out.str();
    }

    inline Transform inverse() const {
        float s = 1.0f / scale;
        Quaternionf r = inverse_unit(rotation);
        Vector3f t = (r * translation) * -s;
        return Transform(t, r, s);
    }
};

// Returns the inverse of the transform.
inline Transform invert(Transform t) {
    return t.inverse();
}

} // NS Math
} // NS Bifrost

// Convenience function that appends a transforms's string representation to an ostream.
inline std::ostream& operator<<(std::ostream& s, Bifrost::Math::Transform t){
    return s << t.to_string();
}

#endif // _BIFROST_MATH_TRANSFORM_H_
