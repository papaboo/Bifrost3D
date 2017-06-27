// Test bed for different unit vector representations.
// See A Survey of Efficient Representations for Independent Unit Vectors, McGuire et al., 2014.
// -----------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// -----------------------------------------------------------------------------------------------

#include <Cogwheel/Math/Distributions.h>
#include <Cogwheel/Math/RNG.h>
#include <Cogwheel/Math/Statistics.h>
#include <Cogwheel/Math/Vector.h>
#include <Cogwheel/Math/Utils.h>

#include <omp.h>
#include <limits.h>

using namespace Cogwheel::Math;

typedef Vector2<unsigned short> Vector2us;
typedef Vector2<short> Vector2s;

inline float sign(float v) { return v >= 0.0f ? +1.0f : -1.0f; }
inline Vector2f sign(Vector2f v) { return Vector2f(sign(v.x), sign(v.y)); }

struct Oct32u {

    Vector2us encoding;

    static Oct32u encode(Vector3f n) {

        // Project the sphere onto the octahedron, and then onto the xy plane.
        Vector2f p = Vector2f(n.x, n.y) / (abs(n.x) + abs(n.y) + abs(n.z));
        
        // Reflect the folds of the lower hemisphere over the diagonals.
        Vector2f p2 = (n.z <= 0.0) ? ((Vector2f(1.0f) - Vector2f(abs(p.y), abs(p.x))) * sign(p)) : p;

        // Fixed point encoding. TODO Current clamp is pointless.
        Oct32u res = { Vector2us(clamp<unsigned short>(unsigned short(p2.x * 32767.5f + 32767.5f), 0u, 65535u),
                                 clamp<unsigned short>(unsigned short(p2.y * 32767.5f + 32767.5f), 0u, 65535u)) };
        return res;
    }

    Vector3f decode() const {
        Vector2f p2 = (Vector2f(encoding) - 32767.5f) / 32767.5f;
        Vector3f n = Vector3f(p2, 1.0f - abs(p2.x) - abs(p2.y));
        if (n.z < 0.0f) {
            float tmp_x = (1.0f - abs(n.y)) * sign(n.x);
            n.y = (1.0f - abs(n.x)) * sign(n.y);
            n.x = tmp_x;
        }
        return normalize(n);
    }
};

struct Oct32s {

    Vector2s encoding;

    static Oct32s encode(Vector3f n) {

        // Project the sphere onto the octahedron, and then onto the xy plane.
        Vector2f p = Vector2f(n.x, n.y) / (abs(n.x) + abs(n.y) + abs(n.z));

        // Reflect the folds of the lower hemisphere over the diagonals.
        Vector2f p2 = (n.z <= 0.0) ? ((Vector2f(1.0f) - Vector2f(abs(p.y), abs(p.x))) * sign(p)) : p;

        // Fixed point encoding.
        Oct32s res = { Vector2s(clamp<int>(int(p2.x * SHRT_MAX), SHRT_MIN, SHRT_MAX),
                                clamp<int>(int(p2.y * SHRT_MAX), SHRT_MIN, SHRT_MAX)) };
        return res;
    }

    Vector3f decode() const {
        Vector2f p2 = Vector2f(encoding) / float(SHRT_MAX);
        Vector3f n = Vector3f(p2, 1.0f - abs(p2.x) - abs(p2.y));
        if (n.z < 0.0f) {
            float tmp_x = (1.0f - abs(n.y)) * sign(n.x);
            n.y = (1.0f - abs(n.x)) * sign(n.y);
            n.x = tmp_x;
        }
        return normalize(n);
    }
};

int main(int argc, char** argv) {
    printf("Unit vector representation tests\n");

    Statistics stats = Statistics(0, 1000000, [](int i) -> double {
        Vector3d normal = normalize((Vector3d)Distributions::Sphere::Sample(RNG::sample02(i)));
        Vector3d decoded_normal = (Vector3d)Oct32u::encode((Vector3f)normal).decode();
        return magnitude(normal - decoded_normal);
    });

    printf("Stats Oct32u:\n");
    printf("  Min: %f\n", stats.minimum);
    printf("  Mean: %f\n", stats.mean);
    printf("  Max: %f\n", stats.maximum);
    printf("  Variance: %f\n", stats.variance);

    stats = Statistics(0, 1000000, [](int i) -> double {
        Vector3d normal = normalize((Vector3d)Distributions::Sphere::Sample(RNG::sample02(i)));
        Vector3d decoded_normal = (Vector3d)Oct32s::encode((Vector3f)normal).decode();
        if (magnitude(normal - decoded_normal) > 0.1f)
            printf("%d: %s -> %s\n", i, normal.to_string().c_str(), decoded_normal.to_string().c_str());
        return magnitude(normal - decoded_normal);
    });

    printf("Stats Oct32s:\n");
    printf("  Min: %f\n", stats.minimum);
    printf("  Mean: %f\n", stats.mean);
    printf("  Max: %f\n", stats.maximum);
    printf("  Variance: %f\n", stats.variance);
}