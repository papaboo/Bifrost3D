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
typedef Vector3<char> Vector3sb;

inline float sign(float v) { return v >= 0.0f ? +1.0f : -1.0f; }
inline Vector2f sign(Vector2f v) { return Vector2f(sign(v.x), sign(v.y)); }

struct XYZ24 {
    Vector3sb encoding;

    static XYZ24 encode(Vector3f n) {
        XYZ24 res = { Vector3sb(char(clamp(n.x, -1.0f, 1.0f) * 127.0f + (n.x < 0 ? -0.5f : 0.5f)),
                                char(clamp(n.y, -1.0f, 1.0f) * 127.0f + (n.y < 0 ? -0.5f : 0.5f)),
                                char(clamp(n.z, -1.0f, 1.0f) * 127.0f + (n.z < 0 ? -0.5f : 0.5f))) };
        return res;
    }

    Vector3f decode() const {
        return normalize(Vector3f(encoding.x / 127.0f, encoding.y / 127.0f, encoding.z / 127.0f));
    }
};

struct Oct32u {

    Vector2us encoding;

    static Oct32u encode(Vector3f n) {

        // Project the sphere onto the octahedron, and then onto the xy plane.
        Vector2f p = Vector2f(n.x, n.y) / (abs(n.x) + abs(n.y) + abs(n.z));
        
        // Reflect the folds of the lower hemisphere over the diagonals.
        Vector2f p2 = n.z < 0 ? (Vector2f(1.0f) - Vector2f(abs(p.y), abs(p.x))) * sign(p) : p;

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
        Vector2f p2 = n.z < 0 ? (Vector2f(1.0f) - Vector2f(abs(p.y), abs(p.x))) * sign(p) : p;

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

//-------------------------------------------------------------------------------------------------
// Encodes a unit vector using an octahedral representation and directx 11's fixed point encoding.
// See A Survey of Efficient Representations for Independent Unit Vectors, McGuire et al., 2014 for how to encode.
// See https://msdn.microsoft.com/en-us/library/windows/desktop/dd607323(v=vs.85).aspx#fixed_point_integer_conversion for how to convert a float to a fixed point snorm.
//-------------------------------------------------------------------------------------------------
struct Oct32s_dx11 {

    Vector2s encoding;

    static Oct32s_dx11 encode(Vector3f n) {

        // Project the sphere onto the octahedron, and then onto the xy plane.
        Vector2f p = Vector2f(n.x, n.y) / (abs(n.x) + abs(n.y) + abs(n.z));

        // Reflect the folds of the lower hemisphere over the diagonals.
        Vector2f p2 = n.z < 0 ? (Vector2f(1.0f) - Vector2f(abs(p.y), abs(p.x))) * sign(p) : p;

        // Fixed point encoding.
        Oct32s_dx11 res = { Vector2s(short(clamp(p2.x, -1.0f, 1.0f) * SHRT_MAX + (p2.x < 0 ? -0.5f : 0.5f)),
                                     short(clamp(p2.y, -1.0f, 1.0f) * SHRT_MAX + (p2.y < 0 ? -0.5f : 0.5f))) };
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

    static Oct32s_dx11 encode_precise(Vector3f n) {
        // Project the sphere onto the octahedron, and then onto the xy plane.
        Vector2f p = Vector2f(n.x, n.y) / (abs(n.x) + abs(n.y) + abs(n.z));

        // Reflect the folds of the lower hemisphere over the diagonals.
        Vector2f p2 = n.z < 0 ? (Vector2f(1.0f) - Vector2f(abs(p.y), abs(p.x))) * sign(p) : p;

        // Fixed point encoding.
        Oct32s_dx11 floored_oct = { Vector2s(short(floor(clamp(p2.x, -1.0f, 1.0f) * SHRT_MAX)),
                                             short(floor(clamp(p2.y, -1.0f, 1.0f) * SHRT_MAX))) };

        Oct32s_dx11 best_representation = floored_oct;
        float lowest_sqrd_mag = squared_magnitude(best_representation.decode() - n);

        auto best_oct_tester = [&](Oct32s_dx11 new_oct) {
            float m = squared_magnitude(new_oct.decode() - n);
            if (m < lowest_sqrd_mag) {
                best_representation = new_oct;
                lowest_sqrd_mag = m;
            }
        };

        Oct32s_dx11 upper_left = { floored_oct.encoding + Vector2s(0, 1)};
        best_oct_tester(upper_left);
        Oct32s_dx11 lower_right = { floored_oct.encoding + Vector2s(1, 0) };
        best_oct_tester(lower_right);
        Oct32s_dx11 upper_right = { floored_oct.encoding + Vector2s(1, 1) };
        best_oct_tester(upper_right);

        return best_representation;
    }
};

typedef Vector3f (*EncodeDecode)(Vector3f);

void test_encoding(const std::string& name, EncodeDecode encode_decode) {
    const int sample_count = 100000;

    Statistics stats = Statistics(0, sample_count, [=](int i) -> double {
        Vector3d normal = normalize((Vector3d)Distributions::Sphere::Sample(RNG::sample02(i)));
        Vector3d decoded_normal = (Vector3d)encode_decode((Vector3f)normal);
        return magnitude(normal - decoded_normal);
    });

    printf("\nStats %s:\n", name.c_str());
    printf("  Min: %f\n", stats.minimum);
    printf("  Mean: %f\n", stats.mean);
    printf("  Max: %f\n", stats.maximum);
    printf("  Variance: %f\n", stats.variance);
}

int main(int argc, char** argv) {
    printf("Unit vector representation tests\n");

    test_encoding("XYZ24", [](Vector3f normal) { return XYZ24::encode(normal).decode(); });
    test_encoding("Oct32u", [](Vector3f normal) { return Oct32u::encode(normal).decode(); });
    test_encoding("Oct32s", [](Vector3f normal) { return Oct32s::encode(normal).decode(); });
    test_encoding("Oct32s_dx11", [](Vector3f normal) { return Oct32s_dx11::encode(normal).decode(); });
    test_encoding("Oct32s_dx11 precise", [](Vector3f normal) { return Oct32s_dx11::encode_precise(normal).decode(); });
}