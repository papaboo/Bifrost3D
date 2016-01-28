// Cogwheel transformations.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_TRANSFORM_H_
#define _COGWHEEL_MATH_TRANSFORM_H_

#include <Cogwheel/Math/Quaternion.h>
#include <Cogwheel/Math/Vector.h>

#include <cstring>
#include <sstream>

namespace Cogwheel {
namespace Math {

/**
* A 3 dimensional affine transform.
* The rotation is given by a quaternion, the position by a vector3 and the transform can be uniformly scaled.
* Only uniform scaling is supported, which leaves out shearing and non-uniform scaling.
*/
struct Transform final {
    //*****************************************************************************
    // Public members
    //*****************************************************************************
    Quaternionf rotation;
    Vector3f translation;
    float scale;

    // Constructor.
    Transform(Vector3f translation = Vector3f::zero(), Quaternionf rotation = Quaternionf::identity(), float scale = 1.0f)
        : rotation(rotation), translation(translation), scale(scale) { }

    static inline Transform identity() {
        return Transform(Vector3f::zero(), Quaternionf::identity(), 1.0f);
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
    inline Vector3f operator*(Vector3f v) const {
        return apply(v);
    }

    // Apply the transform to another transform.
    inline Transform apply(Transform t) const {
        return Transform(this->apply(t.translation), rotation * t.rotation, scale * t.scale);
    }
    // Shorthand overloaded multiplication operator for applying the transform to another transform.
    inline Transform operator*(Transform v) const {
        return apply(v);
    }

    // Rotate transform to look at the target.
    inline void look_at(Vector3f target, Vector3f up = Vector3f::up()) {
        Vector3f direction = normalize(target - translation);
        rotation = Quaternionf::look_in(direction, up);
    }

    const std::string to_string() const {
        std::ostringstream out;
        out << "[translation: " << translation << ", rotation: " << rotation << ", scale: " << scale << "]";
        return out.str();
    }
};

// Returns the inverse of the transform.
inline Transform invert(Transform t) {
    float scale = 1.0f / t.scale;
    Quaternionf rotation = inverse_unit(t.rotation);
    Vector3f translation = (rotation * t.translation) * -scale;
    return Transform(translation, rotation, scale);
}

} // NS Math
} // NS Cogwheel

// Convenience function that appends a transforms's string representation to an ostream.
inline std::ostream& operator<<(std::ostream& s, Cogwheel::Math::Transform t){
    return s << t.to_string();
}

#endif // _COGWHEEL_MATH_TRANSFORM_H_