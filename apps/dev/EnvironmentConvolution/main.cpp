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
#include <chrono>

using namespace Bifrost::Assets;
using namespace Bifrost::Core;
using namespace Bifrost::Input;
using namespace Bifrost::Math;
using namespace Bifrost::Math::Distributions;

enum class ConvolutionType {
    MIS, Light, BSDF, Recursive, Separable, SeparableRecursive
};

bool is_recursive(ConvolutionType convolution) {
    return convolution == ConvolutionType::Recursive || convolution == ConvolutionType::SeparableRecursive;
}

bool is_separable(ConvolutionType convolution) {
    return convolution == ConvolutionType::Separable || convolution == ConvolutionType::SeparableRecursive;
}

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
    ConvolutionType sample_method;
    int sample_count;
    bool headless;

    static Options parse(int argc, char** argv) {
        Options options = { ConvolutionType::MIS, 256, false };

        // Skip the first two arguments, the application name and image path.
        for (int argument = 2; argument < argc; ++argument) {
            if (strcmp(argv[argument], "--mis-sampling") == 0 || strcmp(argv[argument], "-m") == 0)
                options.sample_method = ConvolutionType::MIS;
            else if (strcmp(argv[argument], "--light-sampling") == 0 || strcmp(argv[argument], "-l") == 0)
                options.sample_method = ConvolutionType::Light;
            else if (strcmp(argv[argument], "--bsdf-sampling") == 0 || strcmp(argv[argument], "-b") == 0)
                options.sample_method = ConvolutionType::BSDF;
            else if (strcmp(argv[argument], "--recursive-sampling") == 0 || strcmp(argv[argument], "-r") == 0)
                options.sample_method = ConvolutionType::Recursive;
            else if (strcmp(argv[argument], "--separable-filtering") == 0)
                options.sample_method = ConvolutionType::Separable;
            else if (strcmp(argv[argument], "--separable-recursive-filtering") == 0)
                options.sample_method = ConvolutionType::SeparableRecursive;
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
        case ConvolutionType::MIS: out << "MIS sampling"; break;
        case ConvolutionType::Light: out << "Light sampling"; break;
        case ConvolutionType::BSDF: out << "BSDF sampling"; break;
        case ConvolutionType::Recursive: out << "Recursive sampling"; break;
        case ConvolutionType::Separable: out << "Separable convolution"; break;
        case ConvolutionType::SeparableRecursive: out << "Separable recursive convolution"; break;
        }
        if (sample_method != ConvolutionType::Separable && sample_method != ConvolutionType::SeparableRecursive)
            out << ", " << sample_count << " samples pr pixel";
        out << ".";
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

// ------------------------------------------------------------------------------------------------
// Convolute a latitude longtitude image using a GGX kernel scaled by the cosine of the angle between the 
// direction to the target pixel and the source pixel. The GGX kernel is assumed to be separable.
// Future work:
// * Fix such that it produces the same output as the other filter methods.
//   The initial vertical filter seems to be wrong and summing the contributions of
//   the individual rows diverges from the sum from the other filters.
// * Add parameter for minimal delta cosine between samples to avoid excessive oversampling near the poles.
// * Stabilize floating point computations by summing from the outside and in.
// ------------------------------------------------------------------------------------------------
class LatLongConvoluter final {
private:
    RGB* m_tmp_pixels;
    float* m_sin_thetas;
    int m_max_width, m_max_height;

public:
    LatLongConvoluter(int max_width, int max_height)
        : m_max_width(max_width), m_max_height(max_height)
        , m_tmp_pixels(new RGB[max_width * max_height])
        , m_sin_thetas(new float[max_height]) { }

    ~LatLongConvoluter() {
        delete[] m_sin_thetas;
        delete[] m_tmp_pixels;
    }

    void ggx_kernel(float alpha, int width, int height, RGB* source_pixels, RGB* target_pixels) {
        // Filter using a GGX distribution in a cone around the current pixel.
        const float contribution_threshold = 0.01f;

        // Precompute sin_theta.
        // PBRT p. 728. Account for the non-uniform surface area of the pixels, i.e. the higher density near the poles.
        for (int y = 0; y < height; ++y)
            m_sin_thetas[y] = sinf(PI<float>() * (y + 0.5f) / height);

        auto GGX_D_from_uv = [=](Vector3f up_vector, int x, int y) -> float {
            Vector2f sample_uv = Vector2f((x + 0.5f) / width, (y + 0.5f) / height);
            Vector3f sample_vector = latlong_texcoord_to_direction(sample_uv);
            float cos_theta = fmaxf(0.0f, dot(sample_vector, up_vector));
            float f = Distributions::GGX::D(alpha, cos_theta);
            return f * cos_theta;
        };

        // Filter vertically into the temporary pixel buffer.
        #pragma omp parallel for schedule(dynamic, 16)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {

                Vector2f up_uv = Vector2f((x + 0.5f) / width, (y + 0.5f) / height);
                Vector3f up_vector = latlong_texcoord_to_direction(up_uv);

                float cos_theta = 1;
                float f = Distributions::GGX::D(alpha, cos_theta) * cos_theta;
                float w = f * m_sin_thetas[y];
                RGB radiance = source_pixels[x + y * width] * w;
                float total_weight = w;

                int delta_y = 1;
                while ((f = GGX_D_from_uv(up_vector, x, y + delta_y)) >= contribution_threshold) {
                    if (y - delta_y >= 0) {
                        float w = f * m_sin_thetas[y - delta_y];;
                        radiance += source_pixels[x + (y - delta_y) * width] * w;
                        total_weight += w;
                    }

                    if (y + delta_y < height) {
                        float w = f * m_sin_thetas[y + delta_y];
                        radiance += source_pixels[x + (y + delta_y) * width] * w;
                        total_weight += w;
                    }

                    ++delta_y;
                }

                m_tmp_pixels[x + y * width] = radiance / total_weight;
            }
        }

        // Filter horizontally into the target buffer.
        int half_width = width / 2;
        #pragma omp parallel for schedule(dynamic, 16)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {

                Vector2f up_uv = Vector2f((x + 0.5f) / width, (y + 0.5f) / height);
                Vector3f up_vector = latlong_texcoord_to_direction(up_uv);

                float cos_theta = 1;
                float w = Distributions::GGX::D(alpha, cos_theta) * cos_theta;
                RGB radiance = m_tmp_pixels[x + y * width] * w;
                float total_weight = w;

                auto filter_left = [&](int x) -> int {
                    int low_x = x - 1;
                    low_x = low_x < 0 ? low_x + width : low_x;
                    while (low_x != x && (w = GGX_D_from_uv(up_vector, low_x, y)) >= contribution_threshold) {
                        radiance += m_tmp_pixels[low_x + y * width] * w;
                        total_weight += w;

                        --low_x;
                        low_x = low_x < 0 ? low_x + width : low_x;
                    }

                    return low_x;
                };

                auto filter_right = [&](int x) {
                    int delta_x = 1;
                    while (delta_x < width / 2 && (w = GGX_D_from_uv(up_vector, x + delta_x, y)) >= contribution_threshold) {
                        int high_x = x + delta_x;
                        high_x = high_x >= width ? high_x - width : high_x;
                        radiance += m_tmp_pixels[high_x + y * width] * w;
                        total_weight += w;

                        ++delta_x;
                    }
                };

                int left_x = filter_left(x);
                if (left_x != x)
                    filter_right(x);

                target_pixels[x + y * width] = radiance / total_weight;
            }
        }
    }
};

int initialize(Engine& engine) {
    engine.get_window().set_name("Environment convolution");

    Images::allocate(1);
    Textures::allocate(1);

    std::string file_extension = std::string(g_image_file, g_image_file.length() - 4);
    Image image;
    if (file_extension.compare(".exr") == 0)
        image = TinyExr::load(g_image_file);
    else
        image = StbImageLoader::load(g_image_file);

    if (!image.exists()) {
        printf("Could not load image: %s\n", g_image_file.c_str());
        return 1;
    }

    // Convert image to HDR if needed.
    if (image.get_pixel_format() != PixelFormat::RGB_Float || image.get_gamma() != 1.0f)
        image.change_format(PixelFormat::RGB_Float, 1.0f);

    TextureID texture_ID = Textures::create2D(image.get_ID(), MagnificationFilter::Linear, MinificationFilter::Linear, WrapMode::Repeat, WrapMode::Clamp);
    std::unique_ptr<InfiniteAreaLight> infinite_area_light = nullptr;
    std::vector<LightSample> light_samples = std::vector<LightSample>();
    if (g_options.sample_method == ConvolutionType::Light || g_options.sample_method == ConvolutionType::MIS) {
        infinite_area_light = std::make_unique<InfiniteAreaLight>(texture_ID);
        light_samples.resize(g_options.sample_count * 8);
        #pragma omp parallel for schedule(dynamic, 16)
        for (int s = 0; s < light_samples.size(); ++s)
            light_samples[s] = infinite_area_light->sample(RNG::sample02(s, Vector2ui::zero()));
    }

    std::unique_ptr<LatLongConvoluter> IBL_convolution = nullptr;
    if (is_separable(g_options.sample_method))
        IBL_convolution = std::make_unique<LatLongConvoluter>(image.get_width(), image.get_height());

    auto starttime = std::chrono::system_clock::now();

    printf("\rProgress: %.2f%%", 0.0f);

    std::atomic_int finished_pixel_count;
    finished_pixel_count = 0;
    for (int r = 0; r < g_convoluted_images.size(); ++r) {
        int width = image.get_width(), height = image.get_height();

        g_convoluted_images[r] = Images::create2D("Convoluted image", PixelFormat::RGB_Float, 1.0f, Vector2ui(width, height));
        RGB* target_pixels = g_convoluted_images[r].get_pixels<RGB>();

        // No convolution needed when roughness is 0.
        if (r == 0) {
            #pragma omp parallel for schedule(dynamic, 16)
            for (int i = 0; i < int(image.get_pixel_count()); ++i) {

                int x = i % width;
                int y = i / width;

                target_pixels[x + y * width] = image.get_pixel(Vector2ui(x, y)).rgb();

                if (omp_get_thread_num() == 0)
                    printf("\rProgress: %.2f%%", 0.0f);
            }
            continue;
        }

        float roughness = r / (g_convoluted_images.size() - 1.0f);
        float alpha = roughness * roughness;

        TextureID previous_roughness_tex_ID = TextureID::invalid_UID();
        if (is_recursive(g_options.sample_method)) {
            int prev_r = r - 1;
            previous_roughness_tex_ID = Textures::create2D(g_convoluted_images[prev_r].get_ID(), MagnificationFilter::Linear, MinificationFilter::Linear, WrapMode::Repeat, WrapMode::Clamp);
            float prev_roughness = prev_r / (g_convoluted_images.size() - 1.0f);
            float prev_alpha = prev_roughness * prev_roughness;
            float target_alpha = alpha;
            alpha = sqrt(target_alpha * target_alpha - prev_alpha * prev_alpha);
        }

        if (is_separable(g_options.sample_method)) {
            if (g_options.sample_method == ConvolutionType::Separable)
                IBL_convolution->ggx_kernel(alpha, width, height, image.get_pixels<RGB>(), target_pixels);
            else if (g_options.sample_method == ConvolutionType::SeparableRecursive) {
                Image prev_convoluted_image = Textures::get_image_ID(previous_roughness_tex_ID);
                IBL_convolution->ggx_kernel(alpha, width, height, prev_convoluted_image.get_pixels<RGB>(), target_pixels);
            }
            finished_pixel_count += width * height;
            printf("\rProgress: %.2f%%", 100.0f * float(finished_pixel_count) / (image.get_pixel_count() * (g_convoluted_images.size() - 1)));

        } else {

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
                case ConvolutionType::MIS: {
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
                case ConvolutionType::Light:
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
                case ConvolutionType::BSDF:
                    for (int s = 0; s < g_options.sample_count; ++s) {
                        const GGX::Sample& sample = ggx_samples[(s + bsdf_index_offset) % ggx_samples.size()];
                        Vector2f sample_uv = direction_to_latlong_texcoord(up_rotation * sample.direction);
                        radiance += sample2D(texture_ID, sample_uv).rgb();
                    }
                    break;
                case ConvolutionType::Recursive:
                    for (int s = 0; s < g_options.sample_count; ++s) {
                        const GGX::Sample& sample = ggx_samples[(s + bsdf_index_offset) % ggx_samples.size()];
                        Vector2f sample_uv = direction_to_latlong_texcoord(up_rotation * sample.direction);
                        radiance += sample2D(previous_roughness_tex_ID, sample_uv).rgb();
                    }
                    break;
                }

                radiance /= float(g_options.sample_count);

                target_pixels[x + y * width] = radiance;

                ++finished_pixel_count;
                if (omp_get_thread_num() == 0)
                    printf("\rProgress: %.2f%%", 100.0f * float(finished_pixel_count) / (image.get_pixel_count() * (g_convoluted_images.size() - 1)));
            }
        }

        Textures::destroy(previous_roughness_tex_ID);

        if (g_options.headless)
            output_convoluted_image(g_image_file, g_convoluted_images[r], roughness);
    }

    printf("\rProgress: 100.00%%\n");

    // Print convolution time
    auto endtime = std::chrono::system_clock::now();
    float delta_miliseconds = (float)std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime).count();
    printf("Time to convolute: %.3fseconds\n", delta_miliseconds / 1000);

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
        "     | --separable-filtering: Two-pass convolution using a separable GGX filter.\n"
        "     | --separable-recursive-filtering: Two-pass convolution using a separable GGX filter that uses the previous convoluted image as input.\n"
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