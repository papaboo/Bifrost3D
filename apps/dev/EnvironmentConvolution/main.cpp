// Convolute environment maps with a GGX distribution.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Bifrost/Assets/InfiniteAreaLight.h>
#include <Bifrost/Core/Engine.h>
#include <Bifrost/Core/Window.h>
#include <Bifrost/Input/Keyboard.h>
#include <Bifrost/Math/Quaternion.h>
#include <Bifrost/Math/Distributions.h>
#include <Bifrost/Math/RNG.h>

#include <GLFWDriver.h>
#include <StbImageLoader/StbImageLoader.h>
#include <StbImageWriter/StbImageWriter.h>
#include <TinyExr/TinyExr.h>

#include <omp.h>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <GL/gl.h>
#undef RGB

#include <atomic>
#include <array>
#include <fstream>

using namespace Bifrost::Assets;
using namespace Bifrost::Core;
using namespace Bifrost::Input;
using namespace Bifrost::Math;
using namespace Bifrost::Math::Distributions;

enum class SampleMethod {
    MIS, Light, BSDF, Recursive
};

void output_convoluted_image(const std::string& original_image_file, Image image, float roughness) {
    size_t dot_pos = original_image_file.find_last_of('.');
    std::string file_sans_extension = std::string(original_image_file, 0, dot_pos);
    std::string extension = std::string(original_image_file, dot_pos); 
    if (extension.compare(".jpg") == 0)
        extension = ".bmp";
    std::ostringstream output_file;
    output_file << file_sans_extension << "_roughness_" << roughness << extension;
    if (extension.compare(".exr") == 0)
        TinyExr::store(image.get_ID(), output_file.str());
    else
        StbImageWriter::write(image, output_file.str());
}

struct Options {
    SampleMethod sample_method;
    int sample_count;
    bool headless;

    static Options parse(int argc, char** argv) {
        Options options = { SampleMethod::MIS, 256, false };

        // Skip the first two arguments, the application name and image path.
        for (int argument = 2; argument < argc; ++argument) {
            if (strcmp(argv[argument], "--mis-sampling") == 0 || strcmp(argv[argument], "-m") == 0)
                options.sample_method = SampleMethod::MIS;
            else if (strcmp(argv[argument], "--light-sampling") == 0 || strcmp(argv[argument], "-l") == 0)
                options.sample_method = SampleMethod::Light;
            else if (strcmp(argv[argument], "--bsdf-sampling") == 0 || strcmp(argv[argument], "-b") == 0)
                options.sample_method = SampleMethod::BSDF;
            else if (strcmp(argv[argument], "--recursive-sampling") == 0 || strcmp(argv[argument], "-r") == 0)
                options.sample_method = SampleMethod::Recursive;
            else if (strcmp(argv[argument], "--sample-count") == 0 || strcmp(argv[argument], "-s") == 0)
                options.sample_count = atoi(argv[++argument]);
            else if (strcmp(argv[argument], "--headless") == 0)
                options.headless = true;
        }

        return options;
    }

    std::string to_string() const {
        std::ostringstream out;
        switch (sample_method) {
        case SampleMethod::MIS: out << "MIS sampling, "; break;
        case SampleMethod::Light: out << "Light sampling, "; break;
        case SampleMethod::BSDF: out << "BSDF sampling, "; break;
        case SampleMethod::Recursive: out << "Recursive sampling, "; break;
        }
        out << sample_count << " samples pr pixel.";
        return out.str();
    }
};

std::string g_image_file;
Options g_options;
std::array<Image, 11> g_convoluted_images;

void update(Engine& engine) {
    // Initialize render texture
    static GLuint tex_ID = 0u;
    if (tex_ID == 0u) {
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &tex_ID);
        glBindTexture(GL_TEXTURE_2D, tex_ID);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    }

    const Keyboard* const keyboard = engine.get_keyboard();
    if (keyboard->was_released(Keyboard::Key::P))
        for (int i = 0; i < g_convoluted_images.size(); ++i)
            output_convoluted_image(g_image_file, g_convoluted_images[i], i / (g_convoluted_images.size() - 1.0f));

    static int image_index = 0;
    static int uploaded_image_index = -1;
    if (keyboard->was_released(Keyboard::Key::Left))
        --image_index;
    if (keyboard->was_released(Keyboard::Key::Right))
        ++image_index;
    image_index = clamp(image_index, 0, int(g_convoluted_images.size() - 1));

    if (image_index != uploaded_image_index) {
        float roughness = image_index / (g_convoluted_images.size() - 1.0f);
        std::ostringstream title;
        title << "Environment convolution - Roughness " << roughness;
        engine.get_window().set_name(title.str().c_str());
    }

    { // Update the backbuffer.
        const Window& window = engine.get_window();
        glViewport(0, 0, window.get_width(), window.get_height());

        { // Setup matrices. I really don't need to do this every frame, since they never change.
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(-1, 1, -1.f, 1.f, 1.f, -1.f);

            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
        }

        glBindTexture(GL_TEXTURE_2D, tex_ID);
        if (uploaded_image_index != image_index) {
            const GLint BASE_IMAGE_LEVEL = 0;
            const GLint NO_BORDER = 0;
            Image image = g_convoluted_images[image_index];
            RGB* pixels = image.get_pixels<RGB>();
            RGB* gamma_corrected_pixels = new RGB[image.get_pixel_count()];
            #pragma omp parallel for schedule(dynamic, 16)
            for (int i = 0; i < (int)image.get_pixel_count(); ++i)
                gamma_corrected_pixels[i] = gammacorrect(pixels[i], 1.0f / 2.2f);
            glTexImage2D(GL_TEXTURE_2D, BASE_IMAGE_LEVEL, GL_RGB, image.get_width(), image.get_height(), NO_BORDER, GL_RGB, GL_FLOAT, gamma_corrected_pixels);
            uploaded_image_index = image_index;
        }

        glClear(GL_COLOR_BUFFER_BIT);
        glBegin(GL_QUADS); {

            glTexCoord2f(0.0f, 0.0f);
            glVertex3f(-1.0f, -1.0f, 0.f);

            glTexCoord2f(1.0f, 0.0f);
            glVertex3f(1.0f, -1.0f, 0.f);

            glTexCoord2f(1.0f, 1.0f);
            glVertex3f(1.0f, 1.0f, 0.f);

            glTexCoord2f(0.0f, 1.0f);
            glVertex3f(-1.0f, 1.0f, 0.f);

        } glEnd();
    }
}

int initialize(Engine& engine) {
    engine.get_window().set_name("Environment convolution");

    Images::allocate(1);
    Textures::allocate(1);

    std::string file_extension = std::string(g_image_file, g_image_file.length() - 4);
    Image image;
    if (file_extension.compare(".exr") == 0)
        image = TinyExr::load(g_image_file);
    else
        image = StbImageLoader::load(std::string(g_image_file));

    if (!image.exists()) {
        printf("Could not load image: %s\n", g_image_file.c_str());
        return 1;
    }

    // Convert image to HDR if needed.
    if (image.get_pixel_format() != PixelFormat::RGB_Float || image.get_gamma() != 1.0f) {
        Images::UID old_image_ID = image.get_ID();
        image = ImageUtils::change_format(old_image_ID, PixelFormat::RGB_Float, 1.0f);
        Images::destroy(old_image_ID);
    }

    Textures::UID texture_ID = Textures::create2D(image.get_ID(), MagnificationFilter::Linear, MinificationFilter::Linear, WrapMode::Repeat, WrapMode::Clamp);
    InfiniteAreaLight* infinite_area_light = nullptr;
    std::vector<LightSample> light_samples = std::vector<LightSample>();
    if (g_options.sample_method == SampleMethod::Light || g_options.sample_method == SampleMethod::MIS) {
        infinite_area_light = new InfiniteAreaLight(texture_ID);
        light_samples.resize(g_options.sample_count * 8);
        #pragma omp parallel for schedule(dynamic, 16)
        for (int s = 0; s < light_samples.size(); ++s)
            light_samples[s] = infinite_area_light->sample(RNG::sample02(s, Vector2ui::zero()));
    }

    printf("\rProgress: %.2f%%", 0.0f);

    std::atomic_int finished_pixel_count;
    finished_pixel_count = 0;
    for (int r = 0; r < g_convoluted_images.size(); ++r) {
        int width = image.get_width(), height = image.get_height();

        g_convoluted_images[r] = Images::create2D("Convoluted image", PixelFormat::RGB_Float, 1.0f, Vector2ui(width, height));
        RGB* pixels = g_convoluted_images[r].get_pixels<RGB>();

        // No convolution needed when roughness is 0.
        if (r == 0) {
            #pragma omp parallel for schedule(dynamic, 16)
            for (int i = 0; i < int(image.get_pixel_count()); ++i) {

                int x = i % width;
                int y = i / width;

                pixels[x + y * width] = image.get_pixel(Vector2ui(x, y)).rgb();

                if (omp_get_thread_num() == 0)
                    printf("\rProgress: %.2f%%", 0.0f);
            }
            continue;
        }

        float roughness = r / (g_convoluted_images.size() - 1.0f);
        float alpha = roughness * roughness;

        Textures::UID previous_roughness_tex_ID = Textures::UID::invalid_UID();
        if (g_options.sample_method == SampleMethod::Recursive) {
            previous_roughness_tex_ID = Textures::create2D(g_convoluted_images[r-1].get_ID(), MagnificationFilter::Linear, MinificationFilter::Linear, WrapMode::Repeat, WrapMode::Clamp);
            float prev_roughness = (r - 1.0f) / (g_convoluted_images.size() - 1.0f);
            float prev_alpha = prev_roughness * prev_roughness;
            // ("roughness: %.3f (%.3f), alpha: %.3f (%.3f), recursive alpha: %.5f\n", roughness, prev_roughness, alpha, prev_alpha, alpha * (1.0f - prev_alpha));
            alpha *= 1.0f - prev_alpha;
        }

        std::vector<GGX::Sample> ggx_samples = std::vector<GGX::Sample>();
        ggx_samples.resize(g_options.sample_count * 8);
        #pragma omp parallel for schedule(dynamic, 16)
        for (int s = 0; s < ggx_samples.size(); ++s)
            ggx_samples[s] = GGX::sample(alpha, RNG::sample02(s, Vector2ui::zero()));

        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < int(image.get_pixel_count()); ++i) {

            int x = i % width;
            int y = i / width;

            int bsdf_index_offset = RNG::teschner_hash(x, y) ^ 83492791;
            int light_index_offset = bsdf_index_offset ^ 83492791;

                Vector2f up_uv = Vector2f((x + 0.5f) / width, (y + 0.5f) / height);
            Vector3f up_vector = latlong_texcoord_to_direction(up_uv);
            Quaternionf up_rotation = Quaternionf::look_in(up_vector);

            RGB radiance = RGB::black();

            switch (g_options.sample_method) {
            case SampleMethod::MIS: {
                int bsdf_sample_count = g_options.sample_count / 2;
                int light_sample_count = g_options.sample_count - bsdf_sample_count;

                for (int s = 0; s < light_sample_count; ++s) {
                    const LightSample& sample = light_samples[(s + light_index_offset) % light_samples.size()];
                    if (sample.PDF < 0.000000001f)
                        continue;

                    float cos_theta = fmaxf(dot(sample.direction_to_light, up_vector), 0.0f);
                    float ggx_f = GGX::D(alpha, cos_theta);
                    if (isnan(ggx_f))
                        continue;

                    float mis_weight = RNG::power_heuristic(sample.PDF, GGX::PDF(alpha, cos_theta));
                    radiance += sample.radiance * (mis_weight * ggx_f * cos_theta / sample.PDF);
                }

                for (int s = 0; s < bsdf_sample_count; ++s) {
                    GGX::Sample sample = ggx_samples[(s + bsdf_index_offset) % ggx_samples.size()];
                    if (sample.PDF < 0.000000001f)
                        continue;

                    sample.direction = normalize(up_rotation * sample.direction);
                    float mis_weight = RNG::power_heuristic(sample.PDF, infinite_area_light->PDF(sample.direction));
                    radiance += infinite_area_light->evaluate(sample.direction) * mis_weight;
                }

                // Account for the samples being split evenly between BSDF and light.
                radiance *= 2.0f;

                break;
            }
            case SampleMethod::Light:
                for (int s = 0; s < g_options.sample_count; ++s) {
                    const LightSample& sample = light_samples[(s + light_index_offset) % light_samples.size()];
                    if (sample.PDF < 0.000000001f)
                        continue;

                    float cos_theta = fmaxf(dot(sample.direction_to_light, up_vector), 0.0f);
                    float ggx_f = GGX::D(alpha, cos_theta);
                    if (isnan(ggx_f))
                        continue;

                    radiance += sample.radiance * ggx_f * cos_theta / sample.PDF;
                }
                break;
            case SampleMethod::BSDF:
                for (int s = 0; s < g_options.sample_count; ++s) {
                    const GGX::Sample& sample = ggx_samples[(s + bsdf_index_offset) % ggx_samples.size()];
                    Vector2f sample_uv = direction_to_latlong_texcoord(up_rotation * sample.direction);
                    radiance += sample2D(texture_ID, sample_uv).rgb();
                }
                break;
            case SampleMethod::Recursive:
                for (int s = 0; s < g_options.sample_count; ++s) {
                    const GGX::Sample& sample = ggx_samples[(s + bsdf_index_offset) % ggx_samples.size()];
                    Vector2f sample_uv = direction_to_latlong_texcoord(up_rotation * sample.direction);
                    RGB r = sample2D(previous_roughness_tex_ID, sample_uv).rgb();
                    radiance += gammacorrect(r, 1.0f / 2.2f); // HACK Accumulate in gamma space to reduce fireflies.
                    }
                // Convert back to linear color space.
                radiance = gammacorrect(radiance / float(g_options.sample_count), 2.2f) * float(g_options.sample_count);
                break;
            }

            radiance /= float(g_options.sample_count);

            pixels[x + y * width] = radiance;

            ++finished_pixel_count;
            if (omp_get_thread_num() == 0)
                printf("\rProgress: %.2f%%", 100.0f * float(finished_pixel_count) / (image.get_pixel_count() * (g_convoluted_images.size() - 1)));
        }

        Textures::destroy(previous_roughness_tex_ID);

        if (g_options.headless)
            output_convoluted_image(g_image_file, g_convoluted_images[r], roughness);
    }

    printf("\rProgress: 100.00%%\n");

    // Hook up update callback.
    if (!g_options.headless)
        engine.add_mutating_callback([&] { update(engine); });

    return 0;
}

void print_usage() {
    char* usage =
        "usage EnvironmentConvolution <path/to/environment.ext>\n"
        "  -h | --help: Show command line usage for EnvironmentConvolution.\n"
        "  -s | --sample-count: The number of samples pr pixel.\n"
        "  -m | --mis-sampling: Combine light and bsdf samples by multiple importance sampling.\n"
        "  -l | --light-sampling: Draw samples from the environment.\n"
        "  -b | --bsdf-sampling: Draw samples from the GGX distribution.\n"
        "  -r | --recursive-sampling: Convolute based on the previous convoluted image.\n"
        "     | --headless: Launch without a window and instead output the convoluted images.\n"
        "\n"
        "Keys:\n"
        "  p: Output the images.\n"
        "  Left arrow: View a sharper image.\n"
        "  Right arrow: View a blurrier image.\n";
    printf("%s", usage);
}

int main(int argc, char** argv) {
    printf("Environment convolution\n");

    if (argc == 1 || std::string(argv[1]).compare("-h") == 0 || std::string(argv[1]).compare("--help") == 0) {
        print_usage();
        return 0;
    }

    g_image_file = std::string(argv[1]);

    std::string file_extension = std::string(g_image_file, g_image_file.length() - 4);
    // Check if the file format is supported.
    if (!(file_extension.compare(".bmp") == 0 ||
        file_extension.compare(".exr") == 0 ||
        file_extension.compare(".hdr") == 0 ||
        file_extension.compare(".jpg") == 0 || // Unofficial support. Can't save as jpg.
        file_extension.compare(".png") == 0 ||
        file_extension.compare(".tga") == 0)) {
        printf("Unsupported file format: %s\nSupported formats are: bmp, exr, hdr, png and tga.\n", file_extension.c_str());
        return 2;
    }

    g_options = Options::parse(argc, argv);

    printf("Convolute '%s'\n", argv[1]);
    printf("  %s\n", g_options.to_string().c_str());

    if (g_options.headless) {
        Engine* engine = new Engine("");
        initialize(*engine);
    } else
        GLFWDriver::run(initialize, nullptr);
}