// Test bed for different unit vector representations.
// See A Survey of Efficient Representations for Independent Unit Vectors, McGuire et al., 2014.
// -----------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// -----------------------------------------------------------------------------------------------

#include <Bifrost/Math/Distributions.h>
#include <Bifrost/Math/Half.h>
#include <Bifrost/Math/OctahedralNormal.h>
#include <Bifrost/Math/RNG.h>
#include <Bifrost/Math/Statistics.h>
#include <Bifrost/Math/Vector.h>
#include <Bifrost/Math/Utils.h>

#include <omp.h>
#include <limits.h>

using namespace Bifrost::Math;

using Vector3sb = Vector3<char>;
using Vector3s = Vector3<short>;
using Vector3h = Vector3<half>;

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

struct XYZ32 {
    // Waste's memory, but makes the encoding simpler.
    // That is okay, as this representation is merely for comparison purposes and will not be used.
    Vector3s encoding;

    static XYZ32 encode(Vector3f n) {
        XYZ32 res = { Vector3s(short(clamp(n.x, -1.0f, 1.0f) * 511.0f + (n.x < 0 ? -0.5f : 0.5f)),
                               short(clamp(n.y, -1.0f, 1.0f) * 511.0f + (n.y < 0 ? -0.5f : 0.5f)),
                               short(clamp(n.z, -1.0f, 1.0f) * 255.0f + (n.z < 0 ? -0.5f : 0.5f))) };
        return res;
    }

    Vector3f decode() const {
        return normalize(Vector3f(encoding.x / 511.0f, encoding.y / 511.0f, encoding.z / 255.0f));
    }
};

//-------------------------------------------------------------------------------------------------
// Encodes a unit vector using an octahedral representation and directx 11's fixed point encoding.
// See A Survey of Efficient Representations for Independent Unit Vectors, McGuire et al., 2014 for how to encode.
// See https://msdn.microsoft.com/en-us/library/windows/desktop/dd607323(v=vs.85).aspx#fixed_point_integer_conversion for how to convert a float to a fixed point snorm.
//-------------------------------------------------------------------------------------------------
struct OctahedralUnit64 {

    Vector2i encoding;

    static OctahedralUnit64 encode(Vector3f n) {

        // Project the sphere onto the octahedron, and then onto the xy plane.
        Vector2f p = Vector2f(n.x, n.y) / (abs(n.x) + abs(n.y) + abs(n.z));

        // Reflect the folds of the lower hemisphere over the diagonals.
        Vector2f p2 = n.z < 0 ? (Vector2f(1.0f) - Vector2f(abs(p.y), abs(p.x))) * sign(p) : p;

        // Fixed point encoding.
        OctahedralUnit64 res = { Vector2i(int(clamp(p2.x, -1.0f, 1.0f) * INT_MAX + (p2.x < 0 ? -0.5f : 0.5f)),
            int(clamp(p2.y, -1.0f, 1.0f) * INT_MAX + (p2.y < 0 ? -0.5f : 0.5f))) };
        return res;
    }

    Vector3f decode() const {
        Vector2f p2 = Vector2f(encoding);
        Vector3f n = Vector3f(p2, INT_MAX - abs(p2.x) - abs(p2.y));
        if (n.z < 0.0f) {
            float tmp_x = (INT_MAX - abs(n.y)) * sign(n.x);
            n.y = (INT_MAX - abs(n.x)) * sign(n.y);
            n.x = tmp_x;
        }
        return normalize(n);
    }

    static OctahedralUnit64 encode_precise(Vector3f n) {
        // Project the sphere onto the octahedron, and then onto the xy plane.
        Vector2f p = Vector2f(n.x, n.y) / (abs(n.x) + abs(n.y) + abs(n.z));

        // Reflect the folds of the lower hemisphere over the diagonals.
        Vector2f p2 = n.z < 0 ? (Vector2f(1.0f) - Vector2f(abs(p.y), abs(p.x))) * sign(p) : p;

        // Fixed point encoding.
        OctahedralUnit64 floored_oct = { Vector2i(int(floor(clamp(p2.x, -1.0f, 1.0f) * INT_MAX)),
                                                  int(floor(clamp(p2.y, -1.0f, 1.0f) * INT_MAX))) };

        OctahedralUnit64 best_representation = floored_oct;
        float lowest_sqrd_mag = magnitude_squared(best_representation.decode() - n);

        auto best_oct_tester = [&](OctahedralUnit64 new_oct) {
            float m = magnitude_squared(new_oct.decode() - n);
            if (m < lowest_sqrd_mag) {
                best_representation = new_oct;
                lowest_sqrd_mag = m;
            }
        };

        OctahedralUnit64 upper_left = { floored_oct.encoding + Vector2i(0, 1) };
        best_oct_tester(upper_left);
        OctahedralUnit64 lower_right = { floored_oct.encoding + Vector2i(1, 0) };
        best_oct_tester(lower_right);
        OctahedralUnit64 upper_right = { floored_oct.encoding + Vector2i(1, 1) };
        best_oct_tester(upper_right);

        return best_representation;
    }

    OctahedralNormal to_octahedral_unit_32() const {
        OctahedralNormal res = { Vector2s(encoding.x >> 16, encoding.y >> 16) };
        return res;
    }
};

//-------------------------------------------------------------------------------------------------
// Encodes a normal by storing the sign of the z component in the second most significant bit 
// of the y component. The reason why this works is because the second most significant bit, 
// which is the most significant bit of the exponent, is only ever set when the absolute value of 
// a floating point number is larger than two, which is never the case for normals.
//-------------------------------------------------------------------------------------------------
class ReconstructZ64 {
private:
    float x;
    int y_bitmask_z_sign_y; // Contains the bitmask of the y component and the sign of z encoded in second most significant bit.

public:
    
    static ReconstructZ64 encode(Vector3f n) {
        ReconstructZ64 res;
        res.x = n.x;
        memcpy(&res.y_bitmask_z_sign_y, &n.y, sizeof(float));
        res.y_bitmask_z_sign_y |= n.z < 0.0f ? 0 : (1 << 30);
        return res;
    }

    // Decodes the normal and returns the original normal.
    // Has a max error of approximately 1/2222.
    Vector3f decode() const {
        Vector3f res;
        res.x = x;
        int sign = y_bitmask_z_sign_y & (1 << 30);
        int y_bitmask = y_bitmask_z_sign_y & ~(1 << 30);
        memcpy(&res.y, &y_bitmask, sizeof(float));
        res.z = sqrt(max(1.0f - res.x * res.x - res.y * res.y, 0.0f));
        res.z *= sign == 0 ? -1.0f : 1.0f;
        return res;
    }
};


typedef Vector3f (*EncodeDecode)(Vector3f);

void test_encoding(const std::string& name, EncodeDecode encode_decode) {
    const int sample_count = 100000;

    Statistics<double> stats = Statistics<double>(0, sample_count, [=](int i) -> double {
        Vector3d normal = normalize((Vector3d)Distributions::Sphere::Sample(RNG::sample02(i)));
        Vector3d decoded_normal = (Vector3d)encode_decode((Vector3f)normal);
        return magnitude(normal - decoded_normal);
    });

    printf("\nStats %s:\n", name.c_str());
    printf("  Min: %f\n", stats.minimum);
    printf("  Mean: %f\n", stats.mean());
    printf("  Max: %f\n", stats.maximum);
    printf("  Std dev: %f\n", stats.standard_deviation());
}

int main(int argc, char** argv) {
    printf("Unit vector representation tests\n");

    test_encoding("XYZ24", [](Vector3f normal) { return XYZ24::encode(normal).decode(); });
    test_encoding("XYZ32", [](Vector3f normal) { return XYZ32::encode(normal).decode(); });
    test_encoding("Half3", [](Vector3f normal) { return Vector3f(Vector3h(normal)); });
    test_encoding("ReconstructZ", [](Vector3f normal) { return ReconstructZ64::encode(normal).decode(); });
    test_encoding("Oct32", [](Vector3f normal) { return OctahedralNormal::encode(normal).decode(); });
    test_encoding("Oct32 precise", [](Vector3f normal) { return OctahedralNormal::encode_precise(normal).decode(); });
    test_encoding("Oct64", [](Vector3f normal) { return OctahedralUnit64::encode(normal).decode(); });
    test_encoding("Oct64 precise", [](Vector3f normal) { return OctahedralUnit64::encode_precise(normal).decode(); });
    test_encoding("Oct64 -> Oct32", [](Vector3f normal) { return OctahedralUnit64::encode_precise(normal).to_octahedral_unit_32().decode(); });
}