// Cogwheel transformations.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_TRANSFORM_H_
#define _COGWHEEL_MATH_TRANSFORM_H_

#include <Math/Quaternion.h>
#include <Math/Vector.h>

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
    Quaternionf mRotation;
    Vector3f mTranslation;
    float mScale;

    // Constructor.
    Transform(Vector3f translation = Vector3f::zero(), Quaternionf rotation = Quaternionf::identity(), float scale = 1.0f)
        : mRotation(rotation), mTranslation(translation), mScale(scale) { }

    static inline Transform identity() {
        return Transform(Vector3f::zero(), Quaternionf::identity(), 1.0f);
    }

    //*****************************************************************************
    // Comparison operators.
    //*****************************************************************************
    inline bool operator==(Transform rhs) const {
        return mRotation == rhs.mRotation && mTranslation == rhs.mTranslation && mScale == rhs.mScale;
    }
    inline bool operator!=(Transform rhs) const {
        return mRotation != rhs.mRotation || mTranslation != rhs.mTranslation || mScale != rhs.mScale;
    }

    // Apply the transform to a vector.
    inline Vector3f apply(Vector3f v) const {
        return mTranslation + mRotation * v * mScale;
    }

    // Shorthand overloaded multiplication operator for applying the transform to a vector.
    inline Vector3f operator*(Vector3f v) const {
        return apply(v);
    }

    // Apply the transform to another transform.
    inline Transform apply(Transform t) const {
        float scale = mScale * t.mScale;
        Quaternionf rotation = mRotation * t.mRotation;
        Vector3f translation = this->apply(t.mTranslation);
        return Transform(translation, rotation, scale);
    }
    // Shorthand overloaded multiplication operator for applying the transform to another transform.
    inline Transform operator*(Transform v) const {
        return apply(v);
    }

    // Rotate transform to look at the target.
    inline void lookAt(Vector3f target, Vector3f up = Vector3f::up()) {
        Vector3f direction = normalize(target - mTranslation);
        mRotation = Quaternionf::lookIn(direction, up);
    }

    const std::string toString() const {
        std::ostringstream out;
        out << "[translation: " << mTranslation << ", rotation: " << mRotation << ", scale: " << mScale << "]";
        return out.str();
    }
};

// Returns the inverse of the transform.
inline Transform inverse(Transform t) {
    // TODO Safe inversion of scale, e.g handle NaN (or is it already infinite? Check!)
    float scale = 1.0f / t.mScale;
    Quaternionf rotation = inverse_unit(t.mRotation);
    Vector3f translation = (rotation * t.mTranslation) * -scale;
    return Transform(translation, rotation, scale);
}

} // NS Math
} // NS Cogwheel

// Convinience function that appends a transforms's string representation to an ostream.
inline std::ostream& operator<<(std::ostream& s, Cogwheel::Math::Transform t){
    return s << t.toString();
}

#endif // _COGWHEEL_MATH_TRANSFORM_H_