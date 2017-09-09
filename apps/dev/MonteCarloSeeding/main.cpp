// Test bed for different monte carlo seeding strategies..
// -----------------------------------------------------------------------------------------------
// Copyright (C) 2017, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// -----------------------------------------------------------------------------------------------

#include <Cogwheel/Math/RNG.h>
#include <Cogwheel/Math/Statistics.h>

#include <vector>
#include <limits.h>

using namespace Cogwheel::Math;
using namespace std;

// https://primes.utm.edu/lists/small/10000.txt
const int prime_count = 1230;
static const int primes[1230] =
    { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
    353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
    419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
    467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
    547, 557, 563, 569, 571, 577, 587, 593, 599, 601,
    607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
    661, 673, 677, 683, 691, 701, 709, 719, 727, 733,
    739, 743, 751, 757, 761, 769, 773, 787, 797, 809,
    811, 821, 823, 827, 829, 839, 853, 857, 859, 863,
    877, 881, 883, 887, 907, 911, 919, 929, 937, 941,
    947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013,
    1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069,
    1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151,
    1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223,
    1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291,
    1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373,
    1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451,
    1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511,
    1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583,
    1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657,
    1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733,
    1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811,
    1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889,
    1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987,
    1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053,
    2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129,
    2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213,
    2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287,
    2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357,
    2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423,
    2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531,
    2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617,
    2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687,
    2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741,
    2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819,
    2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903,
    2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999,
    3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079,
    3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181,
    3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257,
    3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331,
    3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413,
    3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511,
    3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571,
    3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643,
    3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727,
    3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821,
    3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907,
    3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989,
    4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057,
    4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139,
    4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219, 4229, 4231,
    4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297,
    4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409,
    4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493,
    4507, 4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583,
    4591, 4597, 4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657,
    4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733, 4751,
    4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831,
    4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937,
    4943, 4951, 4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003,
    5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087,
    5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179,
    5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279,
    5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387,
    5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443,
    5449, 5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521,
    5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639,
    5641, 5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693,
    5701, 5711, 5717, 5737, 5741, 5743, 5749, 5779, 5783, 5791,
    5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851, 5857,
    5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939,
    5953, 5981, 5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053,
    6067, 6073, 6079, 6089, 6091, 6101, 6113, 6121, 6131, 6133,
    6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217, 6221,
    6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 6301,
    6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367,
    6373, 6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473,
    6481, 6491, 6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571,
    6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659, 6661, 6673,
    6679, 6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761,
    6763, 6779, 6781, 6791, 6793, 6803, 6823, 6827, 6829, 6833,
    6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917,
    6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997,
    7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103,
    7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207,
    7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297,
    7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411,
    7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499,
    7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561,
    7573, 7577, 7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643,
    7649, 7669, 7673, 7681, 7687, 7691, 7699, 7703, 7717, 7723,
    7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829,
    7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919,
    7927, 7933, 7937, 7949, 7951, 7963, 7993, 8009, 8011, 8017,
    8039, 8053, 8059, 8069, 8081, 8087, 8089, 8093, 8101, 8111,
    8117, 8123, 8147, 8161, 8167, 8171, 8179, 8191, 8209, 8219,
    8221, 8231, 8233, 8237, 8243, 8263, 8269, 8273, 8287, 8291,
    8293, 8297, 8311, 8317, 8329, 8353, 8363, 8369, 8377, 8387,
    8389, 8419, 8423, 8429, 8431, 8443, 8447, 8461, 8467, 8501,
    8513, 8521, 8527, 8537, 8539, 8543, 8563, 8573, 8581, 8597,
    8599, 8609, 8623, 8627, 8629, 8641, 8647, 8663, 8669, 8677,
    8681, 8689, 8693, 8699, 8707, 8713, 8719, 8731, 8737, 8741,
    8747, 8753, 8761, 8779, 8783, 8803, 8807, 8819, 8821, 8831,
    8837, 8839, 8849, 8861, 8863, 8867, 8887, 8893, 8923, 8929,
    8933, 8941, 8951, 8963, 8969, 8971, 8999, 9001, 9007, 9011,
    9013, 9029, 9041, 9043, 9049, 9059, 9067, 9091, 9103, 9109,
    9127, 9133, 9137, 9151, 9157, 9161, 9173, 9181, 9187, 9199,
    9203, 9209, 9221, 9227, 9239, 9241, 9257, 9277, 9281, 9283,
    9293, 9311, 9319, 9323, 9337, 9341, 9343, 9349, 9371, 9377,
    9391, 9397, 9403, 9413, 9419, 9421, 9431, 9433, 9437, 9439,
    9461, 9463, 9467, 9473, 9479, 9491, 9497, 9511, 9521, 9533,
    9539, 9547, 9551, 9587, 9601, 9613, 9619, 9623, 9629, 9631,
    9643, 9649, 9661, 9677, 9679, 9689, 9697, 9719, 9721, 9733,
    9739, 9743, 9749, 9767, 9769, 9781, 9787, 9791, 9803, 9811,
    9817, 9829, 9833, 9839, 9851, 9857, 9859, 9871, 9883, 9887,
    9901, 9907, 9923, 9929, 9931, 9941, 9949, 9967, 9973, 10007 };

// ------------------------------------------------------------------------------------------------
// Linear congruential random number generator.
// ------------------------------------------------------------------------------------------------
unsigned int reverse_bits(unsigned int n) {
    // Reverse bits of n.
    n = (n << 16) | (n >> 16);
    n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
    n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
    n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
    n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
    return n;
}

// Insert a 0 bit in between each of the 16 low bits of v.
unsigned int part_by_1(unsigned int v) {
    v &= 0x0000ffff;                 // v = ---- ---- ---- ---- fedc ba98 7654 3210
    v = (v ^ (v << 8)) & 0x00ff00ff; // v = ---- ---- fedc ba98 ---- ---- 7654 3210
    v = (v ^ (v << 4)) & 0x0f0f0f0f; // v = ---- fedc ---- ba98 ---- 7654 ---- 3210
    v = (v ^ (v << 2)) & 0x33333333; // v = --fe --dc --ba --98 --76 --54 --32 --10
    v = (v ^ (v << 1)) & 0x55555555; // v = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return v;
}

unsigned int morton_encode(unsigned int x, unsigned int y) {
    return part_by_1(y) | (part_by_1(x) << 1);
}

// ------------------------------------------------------------------------------------------------
// Linear congruential random number generator.
// ------------------------------------------------------------------------------------------------
struct LinearCongruential {
private:
    unsigned int m_state;

public:
    static const unsigned int multiplier = 1664525u;
    static const unsigned int increment = 1013904223u;
    static const unsigned int max = 0xFFFFFFFFu; // uint32 max.

    void seed(unsigned int seed) { m_state = seed; }
    unsigned int get_seed() const { return m_state; }

    unsigned int sample1ui() {
        m_state = multiplier * m_state + increment;
        return m_state;
    }

    float sample1f() {
        const float inv_max = 1.0f / (float(max) + 1.0f);
        return float(sample1ui()) * inv_max;
    }
};

struct SeederStatistics {
    Statistics<double> pixel_stats;
    Statistics<double> neighbourhood_stats;
};

template <typename Seeder>
SeederStatistics seeder_statistics(int width, int height, int sample_count, const Seeder& seeder) {
    vector<Statistics<double> > per_pixel_stats; per_pixel_stats.resize(width * height);
    vector<Statistics<double> > per_neighbourhood_stats; per_neighbourhood_stats.resize(width * height);

    auto sampler = [width, seeder](int x, int y, int sample_count) -> vector<float> {
        vector<float> samples; samples.resize(sample_count);
        LinearCongruential rng;
        for (int s = 0; s < sample_count; ++s) {
            rng.seed(seeder(x, y, s));
            samples[s] = rng.sample1f();
        }
        return samples;
    };

    // Compute error, i.e. the distance in between the samples subtracted by the expected error.
    auto compute_error = [](vector<float>& samples) -> vector<float> {
        sort(samples.begin(), samples.end());

        int sample_count = (int)samples.size();
        float expected_distance = 1.0f / sample_count;
        vector<float> errors; errors.resize(sample_count);
        for (int s = 0; s < sample_count - 1; ++s) {
            float sample_distance = samples[s + 1] - samples[s];
            errors[s] = sample_distance - expected_distance;
        }
        // The error of the largest sample value is computed by wrapping around, e.g. add 1 to the lowest value.
        errors[sample_count - 1] = (samples[0] + 1.0f) - samples[sample_count - 1] - expected_distance;

        // Normalize error by sample count, so the error doesn't just magically drop as the sample count increases.
        for (int e = 0; e < sample_count; ++e)
            errors[e] *= sample_count;

        return errors;
    };

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            int pixel_index = x + y * width;

            // Pr pixel error statistics.
            vector<float> pixel_samples = sampler(x, y, sample_count);
            vector<float> pixel_errors = compute_error(pixel_samples);
            per_pixel_stats[pixel_index] = Statistics<double>(pixel_errors.begin(), pixel_errors.end());

            // Compute error for neighbourhood around pixel. TODO togglable wrap around mode.
            pixel_samples.reserve(5 * sample_count);
            for (float s : sampler((x > 0 ? x : width) - 1, y, sample_count))
                pixel_samples.push_back(s);
            for (float s : sampler(x < width - 1 ? (x + 1) : 0, y, sample_count))
                pixel_samples.push_back(s);
            for (float s : sampler(x, (y > 0 ? y : height) - 1, sample_count))
                pixel_samples.push_back(s);
            for (float s : sampler(x, y < height - 1 ? (y + 1) : 0, sample_count))
                pixel_samples.push_back(s);
            vector<float> neighbour_errors = compute_error(pixel_samples);
            per_neighbourhood_stats[pixel_index] = Statistics<double>(neighbour_errors.begin(), neighbour_errors.end());
        }

    // Merge all stats
    auto pixel_stats = Statistics<double>::merge(per_pixel_stats.begin(), per_pixel_stats.end());
    auto neighbourhood_stats = Statistics<double>::merge(per_neighbourhood_stats.begin(), per_neighbourhood_stats.end());
    SeederStatistics stats = { pixel_stats, neighbourhood_stats };
    return stats;
}

template <typename Seeder>
void test_seeder(const std::string& name, int width, int height, int sample_count, const Seeder& seeder) {
    auto statistics = seeder_statistics(width, height, sample_count, seeder);

    // Output
    printf("%s:\n", name.c_str());
    printf("  Pixel RMS: %f\n", statistics.pixel_stats.rms());
    printf("  Neighbour RMS: %f\n", statistics.neighbourhood_stats.rms());
}

template <typename Seeder>
SeederStatistics seeder_statistics(int width, int height, int sample_count, int dimension_count, const Seeder& seeder) {
    auto statistics = seeder_statistics(width, height, sample_count, seeder);

    for (int i = 1; i < dimension_count; ++i) {
        auto seeder1 = [seeder, i](unsigned int x, unsigned int y, int sample) -> unsigned int {
            LinearCongruential rng; rng.seed(seeder(x, y, sample));
            for (int d = 0; d < i; ++d)
                rng.sample1ui();
            return rng.get_seed();
        };

        auto local_stats = seeder_statistics(width, height, sample_count, seeder1);
        statistics.pixel_stats.merge_with(local_stats.pixel_stats);
        statistics.neighbourhood_stats.merge_with(local_stats.neighbourhood_stats);
    }

    return statistics;
}

template <typename Seeder>
void test_seeder_in_dimensions(const std::string& name, int width, int height, int sample_count, int dimension_count, const Seeder& seeder) {
    auto statistics = seeder_statistics(width, height, sample_count, dimension_count, seeder);

    // Output
    printf("%s with %u dimensions:\n", name.c_str(), dimension_count);
    printf("  Pixel RMS: %f\n", statistics.pixel_stats.rms());
    printf("  Neighbour RMS: %f\n", statistics.neighbourhood_stats.rms());
}

// Distribute a set of ints inside a cube rotated by 45 degrees placed in a grid.
// Fx for radius 1 the pattern looks like this.
// | - | 1 | - |
// | 0 | 2 | 4 |
// | - | 3 | - |
// So far the valid patterns are only valid for the specific rombe that is tested and not when the rombe is moved along x or y.
// A more general approach could fix this.
void build_rombe_pattern(int radius) {
    auto print_grid = [](std::vector<int> grid, int grid_size) {
        for (int y = 0; y < grid_size; ++y) {
            printf("|");
            for (int x = 0; x < grid_size; ++x) {
                int v = grid[x + y * grid_size];
                if (v < 10)
                    printf("  %u |", v);
                else
                    printf(" %u |", v);
            }
            printf("\n");
        }
    };

    int grid_size = radius * 2 + 1;
    int cell_count = grid_size * grid_size;

    int internal_cell_count = radius * 2 + 1;
    for (int r = 0; r < radius; ++r)
        internal_cell_count += 2 * (r * 2 + 1);

    vector<int> grid; grid.resize(cell_count);
    std::vector<int> value_occurence; value_occurence.resize(internal_cell_count);
    for (int my = 0; my < grid_size; ++my)
        for (int mx = 0; mx < grid_size; ++mx) {
            // Clear value occurences.
            for (int i = 0; i < grid_size; ++i)
                value_occurence[i] = 0;

            // Fill grid
            for (int y = 0; y < grid_size; ++y)
                for (int x = 0; x < grid_size; ++x) {
                    int value = (x * mx + y * my) % internal_cell_count;
                    grid[x + y * grid_size] = value;

                    // If the distance as x + y from the center is less than or equal to the radius, then the cell is part of the pattern.
                    int distance = abs(x - radius) + abs(y - radius);
                    if (distance <= radius)
                        value_occurence[value] += 1;
                }

            // Check validity.
            bool pattern_valid = true;
            for (int i = 0; i < grid_size; ++i)
                pattern_valid &= value_occurence[i] == 1;

            if (pattern_valid) {
                printf("mx %u, my %u, is valid: %s\n", mx, my, pattern_valid ? "true" : "false");
                print_grid(grid, grid_size);
                printf("\n");
            }
        }
}

int main(int argc, char** argv) {
    printf("Monte carlo seeding strategies\n");

    int width = 128, height = 128, sample_count = 5;

    // Sampling initialized by jenkins hash.
    auto jenkins_hash = [width](int x, int y, int sample) -> unsigned int {
        return RNG::jenkins_hash(x + y * width) + reverse_bits(sample);
    };
    test_seeder("Jenkins hash", width, height, sample_count, jenkins_hash);
    test_seeder_in_dimensions("Jenkins hash", width, height, sample_count, 4, jenkins_hash);

    // Uniform sampling
    auto uniform_seeder = [width](int x, int y, int sample) -> unsigned int {
        return reverse_bits(sample);
    };
    test_seeder("Uniform", width, height, sample_count, uniform_seeder);

    // Morton encoding seed
    auto morton_seeder = [width](int x, int y, int sample) -> unsigned int {
        unsigned int encoded_index = reverse_bits(morton_encode(x, y));
        return (encoded_index ^ (encoded_index >> 16)) ^ (1013904223 * sample);
    };
    test_seeder("Morton encoding", width, height, sample_count, morton_seeder);

    // Sobol encoding seed
    auto sobol_seeder = [width](int x, int y, int sample) -> unsigned int {
        auto sobol2 = [](unsigned int n, unsigned int scramble) -> unsigned int {
            for (unsigned int v = 1u << 31u; n != 0; n >>= 1u, v ^= v >> 1u)
                if (n & 0x1) scramble ^= v;
            return scramble;
        };

        unsigned int encoded_index = reverse_bits(morton_encode(sobol2(x, 0u), y));
        return (encoded_index ^ (encoded_index >> 16)) + reverse_bits(sample);
    };
    test_seeder("Sobol encoding", width, height, sample_count, sobol_seeder);

    // Follows the pattern.
    // 0.0 | 0.4 | 0.8 | 0.2 | 0.6 | 0.0
    // 0.2 | 0.6 | 0.0 | 0.4 | 0.8 | 0.2
    // 0.4 | 0.8 | 0.2 | 0.6 | 0.0 | 0.4
    // Suboptimal for anything outside the 3x3 cross and quickly breaks down.
    auto optimal3x3 = [](unsigned int x, unsigned int y, int sample) -> unsigned int {
        unsigned int seed = x * 1717986918 + y * 858993459;
        seed = (seed - LinearCongruential::increment) / LinearCongruential::multiplier;
        return seed ^ reverse_bits(sample);
    };
    test_seeder("Optimial3x3 encoding", width, height, sample_count, optimal3x3);
    test_seeder_in_dimensions("Optimial3x3 encoding", width, height, sample_count, 4, optimal3x3);

    // Teschner et al, 2013
    auto teschner_hash = [](unsigned int x, unsigned int y, int sample) -> unsigned int {
        return reverse_bits(RNG::teschner_hash(x, y) ^ sample);
    };
    test_seeder("Teschner hash", width, height, sample_count, teschner_hash);
    test_seeder_in_dimensions("Teschner hash", width, height, sample_count, 4, teschner_hash);

    build_rombe_pattern(3);
}