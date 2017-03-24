// Convolute environment maps with a GGX distribution.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Cogwheel/Assets/LatLongDistribution.h>
#include <Cogwheel/Math/Quaternion.h>
#include <Cogwheel/Math/RNG.h>

#include <StbImageLoader/StbImageLoader.h>
#include <StbImageWriter/StbImageWriter.h>

#include <fstream>

using namespace Cogwheel;
using namespace Cogwheel::Assets;
using namespace Cogwheel::Math;

//==============================================================================
// GGX distribution.
//==============================================================================
namespace GGX {

struct Sample{
    Vector3f direction;
    float PDF;
};

inline float D(float alpha, float abs_cos_theta) {
    float alpha_sqrd = alpha * alpha;
    float cos_theta_sqrd = abs_cos_theta * abs_cos_theta;
    float tan_theta_sqrd = fmaxf(1.0f - cos_theta_sqrd, 0.0f) / cos_theta_sqrd;
    float cos_theta_cubed = cos_theta_sqrd * cos_theta_sqrd;
    float foo = alpha_sqrd + tan_theta_sqrd; // No idea what to call this.
    return alpha_sqrd / (PI<float>() * cos_theta_cubed * foo * foo);
}

inline float PDF(float alpha, float abs_cos_theta) {
    return D(alpha, abs_cos_theta) * abs_cos_theta;
}

inline Sample sample(float alpha, Vector2f random_sample) {
    float phi = random_sample.y * (2.0f * PI<float>());

    float tan_theta_sqrd = alpha * alpha * random_sample.x / (1.0f - random_sample.x);
    float cos_theta = 1.0f / sqrt(1.0f + tan_theta_sqrd);

    float r = sqrt(fmaxf(1.0f - cos_theta * cos_theta, 0.0f));

    Sample res;
    res.direction = Vector3f(cos(phi) * r, sin(phi) * r, cos_theta);
    res.PDF = PDF(alpha, cos_theta); // We have to be able to inline this to reuse some temporaries.
    return res;
}

} // NS GGX

enum class SampleMethod {
    MIS, Light, BSDF
};

struct Options {
    SampleMethod sample_method;
    int sample_count;

    static Options parse(int argc, char** argv) {
        Options options = { SampleMethod::BSDF, 256 };

        // Skip the first two arguments, the application name and image path.
        for (int argument = 2; argument < argc; ++argument) {
            if (strcmp(argv[argument], "--mis-sampling") == 0 || strcmp(argv[argument], "-m") == 0)
                options.sample_method = SampleMethod::MIS;
            else if (strcmp(argv[argument], "--light-sampling") == 0 || strcmp(argv[argument], "-l") == 0)
                options.sample_method = SampleMethod::Light;
            else if (strcmp(argv[argument], "--bsdf-sampling") == 0 || strcmp(argv[argument], "-b") == 0)
                options.sample_method = SampleMethod::BSDF;
        }

        return options;
    }

    std::string to_string() const {
        std::ostringstream out;
        switch (sample_method) {
        case SampleMethod::MIS: out << "MIS sampling, "; break;
        case SampleMethod::Light: out << "Light sampling, "; break;
        case SampleMethod::BSDF: out << "BSDF sampling, "; break;
        }
        out << sample_count << " samples pr pixel.";
        return out.str();
    }
};

void print_usage() {
    char* usage =
        "usage EnvironmentConvolution <path/to/environment.ext>:\n"
        "  -h | --help: Show command line usage for EnvironmentConvolution.\n"
        "  -s | --sample-count. The number of samples pr pixel.\n"
        "  -m | --mis-sampling. Combine light and bsdf samples by multiple importance sampling.\n"
        "  -l | --light-sampling. Draw samples from the environment.\n"
        "  -b | --bsdf-sampling. Draw samples from the GGX distribution.\n";
    printf("%s", usage);
}

int main(int argc, char** argv) {
    printf("Environment convolution\n");

    if (argc == 1 || std::string(argv[1]).compare("-h") == 0 || std::string(argv[1]).compare("--help") == 0) {
        print_usage();
        return 0;
    }

    Options options = Options::parse(argc, argv);

    printf("Convolute '%s'\n", argv[1]);
    printf("  %s\n", options.to_string().c_str());

    Images::allocate(1);
    Textures::allocate(1);

    Image image = StbImageLoader::load(std::string(argv[1]));

    Textures::UID texture_ID = Textures::create2D(image.get_ID(), MagnificationFilter::Linear, MinificationFilter::Linear, WrapMode::Repeat, WrapMode::Clamp);
    LatLongDistribution* infinite_area_light = nullptr;
    if (options.sample_method == SampleMethod::Light)
        infinite_area_light = new LatLongDistribution(texture_ID);

    Image output = Images::create("Convoluted image", PixelFormat::RGB24, 2.2f, Vector2ui(image.get_width(), image.get_height())); // TODO Wrong gamma

    float alpha = 0.5f;
    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < int(image.get_height()); ++y)
        for (unsigned int x = 0; x < image.get_width(); ++x) {

            Vector2f up_uv = Vector2f((x + 0.5f) / image.get_width(), (y + 0.5f) / image.get_height());
            Vector3f up_vector = latlong_texcoord_to_direction(up_uv);
            Quaternionf up_rotation = Quaternionf::look_in(up_vector);

            // TODO We can precompute the unrotated GGX samples and the area_light samples. Does that speed anything up? Will there be correlation artefacts?
            //      Perhaps just draw samplecount * 16 and reuse different permutations of them.
            // TODO What is faster for rotation? A matrix or quaternion?
            const int SAMPLE_COUNT = 256;
            RGB radiance = RGB::black();
            
            switch (options.sample_method) {
            case SampleMethod::MIS:
                // TODO MIS.
                break;
            case SampleMethod::Light:
                for (int s = 0; s < SAMPLE_COUNT; ++s) {
                    LightSample sample = infinite_area_light->sample(RNG::sample02(s));
                    Vector3f local_direction = inverse_unit(up_rotation) * sample.direction_to_light;
                    float ggx_f = GGX::D(alpha, local_direction.z);
                    float cos_theta = fmaxf(local_direction.z, 0.0f);
                    radiance += sample.radiance * ggx_f * cos_theta / sample.PDF;
                }
                break;
            case SampleMethod::BSDF:
                for (int s = 0; s < SAMPLE_COUNT; ++s) {
                    GGX::Sample sample = GGX::sample(alpha, RNG::sample02(s));
                    sample.direction = up_rotation * sample.direction;
                    Vector2f sample_uv = direction_to_latlong_texcoord(sample.direction);
                    radiance += sample2D(texture_ID, sample_uv).rgb();
                }
                break;
            }

            radiance /= SAMPLE_COUNT;

            output.set_pixel(RGBA(radiance), Vector2ui(x, y));
        }

    // TODO Remove!
    if (options.sample_method == SampleMethod::BSDF)
        StbImageWriter::write("C:/Users/asger/Desktop/output_BSDF.png", output);
    else if (options.sample_method == SampleMethod::Light)
        StbImageWriter::write("C:/Users/asger/Desktop/output_light.png", output);
    else
        StbImageWriter::write("C:/Users/asger/Desktop/output_MIS.png", output);

    return 0;
}
