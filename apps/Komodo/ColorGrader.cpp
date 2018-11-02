// Komodo image color grader.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <ColorGrader.h>
#include <Utils.h>

#include <Cogwheel/Core/Engine.h>
#include <Cogwheel/Math/CameraEffects.h>
#include <ImageOperations/Blur.h>
#include <ImageOperations/Exposure.h>

#include <AntTweakBar/AntTweakBar.h>

#include <array>

using namespace Cogwheel::Assets;
using namespace Cogwheel::Core;
using namespace Cogwheel::Math;
using namespace ImageOperations;

using Vector4uc = Vector4<unsigned char>;

struct ColorGrader::Implementation final {

    // --------------------------------------------------------------------------------------------
    // Members
    // --------------------------------------------------------------------------------------------
    enum class Operator { Linear, Reinhard, FilmicAlu, Uncharted2, Unreal4 };
    Operator m_operator = Operator::Unreal4;
    Image m_input;
    RGB* m_tonemapped_pixels;
    std::string m_output_path;

    GLuint m_tex_ID = 0u;
    bool m_upload_image = true;

    // Exposure members and helpers
    float m_exposure_bias = 0.0f;
    float luminance_scale() { return exp2(m_exposure_bias); }
    float scene_key() { return 0.5f / luminance_scale(); }

    struct {
        bool enabled = false;
        float threshold = 1.5f;
        float support = 0.05f;
    } m_bloom;

    struct {
        const int size = 100;

        float min_percentage = 0.8f;
        float max_percentage = 0.95f;
        float min_log_luminance = -8;
        float max_log_luminance = 4;
        std::array<unsigned int, 100> histogram;
        bool visualize = false;

        GLuint texture_ID;
        Vector4uc* texture_pixels;
    } m_histogram;

    // Tonemapping members
    float m_reinhard_whitepoint = 3.0f;

    CameraEffects::Uncharted2Settings m_uncharted2 = CameraEffects::Uncharted2Settings::default();
    CameraEffects::FilmicSettings m_unreal4 = CameraEffects::FilmicSettings::default();

    TwBar* m_gui = nullptr;

    // --------------------------------------------------------------------------------------------
    // Histogram helpers
    // --------------------------------------------------------------------------------------------

    void compute_histogram_high_low_normalized_index(float& low_normalized_index, float& high_normalized_index) {
        // Compute log luminance values from histogram.
        int min_pixel_count = int(m_input.get_pixel_count() * m_histogram.min_percentage);
        int max_pixel_count = int(m_input.get_pixel_count() * m_histogram.max_percentage);

        int previous_pixel_count = 0;
        int low_index = 0;
        for (; low_index < m_histogram.histogram.size(); ++low_index) {
            int current_pixel_count = previous_pixel_count + m_histogram.histogram[low_index];
            if (previous_pixel_count <= min_pixel_count && min_pixel_count <= current_pixel_count) {
                float decimal_i = float(low_index) + inverse_lerp(float(previous_pixel_count), float(current_pixel_count), float(min_pixel_count));
                low_normalized_index = decimal_i / float(m_histogram.histogram.size());
                break;
            }

            previous_pixel_count = current_pixel_count;
        }

        for (int high_index = low_index; high_index < m_histogram.histogram.size(); ++high_index) {
            int current_pixel_count = previous_pixel_count + m_histogram.histogram[high_index];
            if (previous_pixel_count <= max_pixel_count && max_pixel_count <= current_pixel_count) {
                float decimal_i = float(high_index) + inverse_lerp(float(previous_pixel_count), float(current_pixel_count), float(max_pixel_count));
                high_normalized_index = decimal_i / float(m_histogram.histogram.size());
                break;
            }

            previous_pixel_count = current_pixel_count;
        }
    }

    static inline float compute_luminance_from_histogram_position(float normalized_index, float min_log_luminance, float max_log_luminance) {
        return exp2(lerp(min_log_luminance, max_log_luminance, normalized_index));
    }

    // Unreal 4 PostProcessHistogramCommon.ush::ComputeAverageLuminaneWithoutOutlier
    template <int histogram_size>
    static inline float compute_average_luminance_without_outlier(std::array<unsigned int, histogram_size>& histogram, float min_fraction_sum, float max_fraction_sum,
                                                                  float min_log_luminance, float max_log_luminance) {
        Vector2f sum = Vector2f::zero();

        for (int i = 0; i < histogram_size; ++i) {
            float bucket_count = float(histogram[i]);

            // remove outlier at lower end
            float sub = min(bucket_count, min_fraction_sum);
            bucket_count -= sub;
            min_fraction_sum -= sub;
            max_fraction_sum -= sub;

            // remove outlier at upper end
            bucket_count = min(bucket_count, max_fraction_sum);
            max_fraction_sum -= bucket_count;

            float luminance_at_bucket = compute_luminance_from_histogram_position((i + 0.5f) / float(histogram_size), min_log_luminance, max_log_luminance);

            sum += Vector2f(luminance_at_bucket * bucket_count, bucket_count);
        }

        return sum.x / max(0.0001f, sum.y);
    }

    void fill_histogram_texture(std::array<unsigned int, 100>& histogram, Vector4uc* pixels) {
        // Assumes there are size x size pixels
        unsigned int max_value = 0;
        for (unsigned int v : histogram)
            max_value = max(v, max_value);

        float low_normalized_index, high_normalized_index;
        compute_histogram_high_low_normalized_index(low_normalized_index, high_normalized_index);
        unsigned int low_index = unsigned int(floor(low_normalized_index * histogram.size()));
        unsigned int high_index = unsigned int(ceil(high_normalized_index * histogram.size()));

        unsigned int size = unsigned int(histogram.size());
        for (unsigned int x = 0; x < size; ++x) {
            unsigned int v = histogram[x];
            float normalized_cutoff = v / float(max_value);
            unsigned int cutoff = unsigned int(normalized_cutoff * size);

            bool in_range = low_index <= x && x <= high_index;
            Vector4uc background_color = { 64, 64, 64, 255 };
            background_color.y += in_range ? background_color.y / 2 : 0;
            Vector4uc column_color = { 128, 128, 128, 255 };
            column_color.y += in_range ? column_color.y / 2 : 0;
            for (unsigned int y = 0; y < size; ++y)
                *pixels++ = y <= cutoff ? column_color : background_color;
        }
    }

    // --------------------------------------------------------------------------------------------
    // Print help message.
    // --------------------------------------------------------------------------------------------
    static void print_usage() {
        static char* usage =
            "usage Komodo Tonemapping:\n"
            "  -h | --help: Show command line usage for Komodo Tonemapping.\n"
            "     | --linear: No tonemapping.\n"
            "     | --reinhard: Apply Reinhard tonemapper.\n"
            "     | --uncharted2: Apply Uncharted 2 filmic tonemapper.\n"
            "     | --unreal4: Apply Unreal Engine 4 filmic tonemapper.\n"
            "     | --input <path>: Path to the image to be tonemapped.\n"
            "     | --output <path>: Path to where to store the final image.\n";

        printf("%s", usage);
    }

    // --------------------------------------------------------------------------------------------
    // Parse command line arguments.
    // --------------------------------------------------------------------------------------------
    std::string parse_arguments(std::vector<char*> args) {
        std::string input_path;
        for (int i = 0; i < args.size(); ++i) {
            std::string arg = args[i];
            if (arg.compare("--linear") == 0)
                m_operator = Operator::Linear;
            else if (arg.compare("--reinhard") == 0)
                m_operator = Operator::Reinhard;
            else if (arg.compare("--uncharted2") == 0)
                m_operator = Operator::Uncharted2;
            else if (arg.compare("--unreal4") == 0)
                m_operator = Operator::Unreal4;
            else if (arg.compare("--input") == 0)
                input_path = args[++i];
            else if (arg.compare("--output") == 0)
                m_output_path = args[++i];
            else
                printf("Unknown argument: %s\n", args[i]);
        }

        return input_path;
    }

#define WRAP_ANT_PROPERTY(member_name, T) \
[](const void* input_data, void* client_data) { \
    ColorGrader::Implementation* data = (ColorGrader::Implementation*)client_data; \
    data->member_name = *(T*)input_data; \
    data->m_upload_image = true; \
}, \
[](void* value, void* client_data) { \
    *(T*)value = ((ColorGrader::Implementation*)client_data)->member_name; \
}

    // --------------------------------------------------------------------------------------------
    // GUI.
    // --------------------------------------------------------------------------------------------
    TwBar* setup_gui() {
        TwInit(TW_OPENGL, nullptr);
        TwDefine("TW_HELP visible=false");  // Ant help bar is hidden.

        TwBar* bar = TwNewBar("ColorGrader");

        { // Exposure mapping

            TwAddVarCB(bar, "Bias", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_exposure_bias, float), this, "step=0.1 group=Exposure");
            TwAddVarCB(bar, "Scene key", TW_TYPE_FLOAT, nullptr, [](void* value, void* client_data) {
                *(float*)value = ((ColorGrader::Implementation*)client_data)->scene_key();
            }, this, "step=0.001 group=Exposure");
            TwAddVarCB(bar, "Luminance scale", TW_TYPE_FLOAT, nullptr, [](void* value, void* client_data) {
                *(float*)value = ((ColorGrader::Implementation*)client_data)->luminance_scale();
            }, this, "step=0.001 group=Exposure");

            auto reinhard_auto_exposure = [](void* client_data) {
                ColorGrader::Implementation* data = (ColorGrader::Implementation*)client_data;
                float scene_key = Exposure::log_average_luminance(data->m_input.get_ID());
                float linear_exposure = 0.5f / scene_key;
                data->m_exposure_bias = log2(linear_exposure);
                data->m_upload_image = true;
            };
            TwAddButton(bar, "Set from log-average", reinhard_auto_exposure, this, "group=Exposure");

            auto auto_geometric_mean = [](void* client_data) {
                ColorGrader::Implementation* data = (ColorGrader::Implementation*)client_data;
                float log_average_luminance = Exposure::log_average_luminance(data->m_input.get_ID());
                float key_value = 1.03f - (2.0f / (2 + log10(log_average_luminance + 1)));
                float linear_exposure = key_value / log_average_luminance;
                data->m_exposure_bias = log2(max(linear_exposure, 0.0001f));

                data->m_upload_image = true;
            };
            TwAddButton(bar, "Set from geometric mean", auto_geometric_mean, this, "group=Exposure");

            { // Histogram
                m_histogram.texture_pixels = new Vector4uc[m_histogram.size * m_histogram.size];

                auto histogram_auto_exposure = [](void* client_data) {
                    ColorGrader::Implementation* data = (ColorGrader::Implementation*)client_data;
                    auto& histogram = data->m_histogram;

                    std::fill(histogram.histogram.begin(), histogram.histogram.end(), 0u);
                    Exposure::log_luminance_histogram(data->m_input.get_ID(), histogram.min_log_luminance, histogram.max_log_luminance, 
                                                                       histogram.histogram.begin(), histogram.histogram.end());

                    // Compute linear exposure from histogram.
                    float min_pixel_count = data->m_input.get_pixel_count() * histogram.min_percentage;
                    float max_pixel_count = data->m_input.get_pixel_count() * histogram.max_percentage;
                    float image_average_luminance = compute_average_luminance_without_outlier(histogram.histogram, min_pixel_count, max_pixel_count, 
                                                                                              histogram.min_log_luminance, histogram.max_log_luminance);
                    float linear_exposure = 1.0f / image_average_luminance;
                    
                    data->m_exposure_bias = log2(linear_exposure);

                    data->m_upload_image = true;
                };
                TwAddButton(bar, "Set from histogram", histogram_auto_exposure, this, "group=Histogram");

                TwAddVarCB(bar, "Min percentage", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_histogram.min_percentage, float), this, "step=0.05 group=Histogram");
                TwAddVarCB(bar, "Max percentage", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_histogram.max_percentage, float), this, "step=0.05 group=Histogram");
                TwAddVarCB(bar, "Min log luminance", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_histogram.min_log_luminance, float), this, "step=0.1 group=Histogram");
                TwAddVarCB(bar, "Max log luminance", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_histogram.max_log_luminance, float), this, "step=0.1 group=Histogram");
                TwAddVarCB(bar, "Visualize", TW_TYPE_BOOLCPP, WRAP_ANT_PROPERTY(m_histogram.visualize, bool), this, "group=Histogram");

                TwDefine("ColorGrader/Histogram group='Exposure'");

                histogram_auto_exposure(this);
            }
        }

        { // Bloom
            TwAddVarCB(bar, "Enabled", TW_TYPE_BOOLCPP, WRAP_ANT_PROPERTY(m_bloom.enabled, bool), this, "group=Bloom");
            TwAddVarCB(bar, "Threshold", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_bloom.threshold, float), this, "step=0.1 group=Bloom");
            TwAddVarCB(bar, "Support", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_bloom.support, float), this, "step=0.01 group=Bloom");
        }

        { // Tonemapping
            TwEnumVal operators[] = { { int(Operator::Linear), "Linear" },
                                      { int(Operator::Reinhard), "Reinhard" }, 
                                      { int(Operator::FilmicAlu), "FilmicAlu" },
                                      { int(Operator::Uncharted2), "Uncharted2" },
                                      { int(Operator::Unreal4), "Unreal4" } };
            TwType AntOperatorEnum = TwDefineEnum("Operators", operators, 5);

            auto set_m_operator = [](const void* input_data, void* client_data) {
                ColorGrader::Implementation* data = (ColorGrader::Implementation*)client_data;
                data->m_operator = *(Operator*)input_data;
                data->m_upload_image = true;

                auto show_reinhard = std::string("ColorGrader/Reinhard visible=") + (data->m_operator == Operator::Reinhard ? "true" : "false");
                TwDefine(show_reinhard.c_str());

                auto show_uncharted2 = std::string("ColorGrader/Uncharted2 visible=") + (data->m_operator == Operator::Uncharted2 ? "true" : "false");
                TwDefine(show_uncharted2.c_str());

                auto show_unreal4 = std::string("ColorGrader/Unreal4 visible=") + (data->m_operator == Operator::Unreal4 ? "true" : "false");
                TwDefine(show_unreal4.c_str());
            };
            auto get_m_operator = [](void* value, void* client_data) {
                *(Operator*)value = ((ColorGrader::Implementation*)client_data)->m_operator;
            };
            TwAddVarCB(bar, "Operator", AntOperatorEnum, set_m_operator, get_m_operator, this, "group='Tonemapping'");

            { // Reinhard
                TwAddVarCB(bar, "White point", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_reinhard_whitepoint, float), this, "min=0 step=0.1 group='Reinhard'");

                TwDefine("ColorGrader/Reinhard group='Tonemapping'");
            }

            { // Uncharted 2 filmic
                TwAddVarCB(bar, "Shoulder strength", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_uncharted2.shoulder_strength, float), this, "min=0 step=0.1 group=Uncharted2");
                TwAddVarCB(bar, "Linear strength", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_uncharted2.linear_strength, float), this, "min=0 step=0.1 group=Uncharted2");
                TwAddVarCB(bar, "Linear angle", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_uncharted2.linear_angle, float), this, "min=0 step=0.1 group=Uncharted2");
                TwAddVarCB(bar, "Toe strength", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_uncharted2.toe_strength, float), this, "min=0 step=0.1 group=Uncharted2");
                TwAddVarCB(bar, "Toe numerator", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_uncharted2.toe_numerator, float), this, "min=0 step=0.1 group=Uncharted2");
                TwAddVarCB(bar, "Toe denominator", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_uncharted2.toe_denominator, float), this, "min=0 step=0.1 group=Uncharted2");
                TwAddVarCB(bar, "Linear white", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_uncharted2.linear_white, float), this, "min=0 step=0.1 group=Uncharted2");

                TwDefine("ColorGrader/Uncharted2 group='Tonemapping' label='Uncharted 2'");
            }

            { // Unreal 4 filmic
                { // Presets
                    enum class Presets { None, ACES, Uncharted2, HP, Legacy };
                    TwEnumVal ant_presets[] = { { int(Presets::None), "Select preset" },
                                                { int(Presets::ACES), "ACES" },
                                                { int(Presets::Uncharted2), "Uncharted2" },
                                                { int(Presets::HP), "HP" },
                                                { int(Presets::Legacy), "Legacy" } };
                    TwType AntPresetsEnum = TwDefineEnum("Presets", ant_presets, 6);

                    auto set_preset = [](const void* input_data, void* client_data) {
                        ColorGrader::Implementation* data = (ColorGrader::Implementation*)client_data;
                        Presets preset = *(Presets*)input_data;

                        switch (preset) {
                        case Presets::None:
                            break; // Do nothing
                        case Presets::ACES:
                            data->m_unreal4 = CameraEffects::FilmicSettings::ACES();
                            break;
                        case Presets::Uncharted2:
                            data->m_unreal4 = CameraEffects::FilmicSettings::uncharted2();
                            break;
                        case Presets::HP:
                            data->m_unreal4 = CameraEffects::FilmicSettings::HP();
                            break;
                        case Presets::Legacy:
                            data->m_unreal4 = CameraEffects::FilmicSettings::legacy();
                            break;
                        default:
                            data->m_unreal4 = CameraEffects::FilmicSettings::default();
                            break;
                        }

                        data->m_upload_image = true;
                    };
                    auto get_preset = [](void* value, void* client_data) { *(Presets*)value = Presets::None; };
                    TwAddVarCB(bar, "Presets", AntPresetsEnum, set_preset, get_preset, this, "group='Unreal4'");
                }

                TwAddVarCB(bar, "Black clip", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_unreal4.black_clip, float), this, "step=0.1 group=Unreal4");
                TwAddVarCB(bar, "Toe", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_unreal4.toe, float), this, "step=0.1 group=Unreal4");
                TwAddVarCB(bar, "Slope", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_unreal4.slope, float), this, "step=0.1 group=Unreal4");
                TwAddVarCB(bar, "Shoulder", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_unreal4.shoulder, float), this, "step=0.1 group=Unreal4");
                TwAddVarCB(bar, "White clip", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_unreal4.white_clip, float), this, "step=0.1 group=Unreal4");

                TwDefine("ColorGrader/Unreal4 group='Tonemapping'");
            }

            auto tonemapper = Operator::Unreal4;
            set_m_operator(&tonemapper, this);
        }
        
        // Save button.
        if (m_output_path.size() != 0) {
            auto save_image = [](void* client_data) {
                // Tonemap and store the image
                ColorGrader::Implementation* data = (ColorGrader::Implementation*)client_data;
                Vector2ui size = { data->m_input.get_width(), data->m_input.get_height() };
                Image output_image = Images::create2D("tonemapped_" + data->m_input.get_name(), PixelFormat::RGB_Float, 2.2f, size);
                RGB* pixels = output_image.get_pixels<RGB>();
                data->color_grade_image(data->m_input, pixels);
                store_image(output_image, data->m_output_path);
                Images::destroy(output_image.get_ID());
            };
            TwAddButton(bar, "Save", save_image, this, "");
        }

        return bar;
    }

    // --------------------------------------------------------------------------------------------
    // Constructor.
    // --------------------------------------------------------------------------------------------
    Implementation(std::vector<char*> args, Cogwheel::Core::Engine& engine) {

        if (args.size() == 0 || std::string(args[0]).compare("-h") == 0 || std::string(args[0]).compare("--help") == 0) {
            print_usage();
            return;
        }

        // Parse arguments
        std::string input_path = parse_arguments(args);

        // Load input image
        m_input = load_image(input_path);
        if (!m_input.exists()) {
            printf("  error: Could not load image at '%s'\n", input_path.c_str());
            m_input = create_error_image();
        }
        engine.get_window().set_name("Komodo - " + m_input.get_name());

        bool headless = engine.get_window().get_width() == 0 && engine.get_window().get_height() == 0;
        if (headless) {
            if (m_output_path.size() != 0) {
                // Color grade and store the image
                Vector2ui size = { m_input.get_width(), m_input.get_height() };
                Image output_image = Images::create2D("tonemapped_" + m_input.get_name(), PixelFormat::RGB_Float, 2.2f, size);
                RGB* pixels = output_image.get_pixels<RGB>();
                color_grade_image(m_input, pixels);
                store_image(output_image, m_output_path);
            }
        } else {
            auto create_texture = []() -> GLuint {
                GLuint tex_ID = 0;
                glGenTextures(1, &tex_ID);
                glBindTexture(GL_TEXTURE_2D, tex_ID);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                return tex_ID;
            };

            glEnable(GL_TEXTURE_2D);
            m_tex_ID = create_texture();
            m_histogram.texture_ID = create_texture();

            m_tonemapped_pixels = new RGB[m_input.get_pixel_count()];
            m_gui = setup_gui();
            engine.add_mutating_callback(ColorGrader::Implementation::update, this);
        }
    }

    // --------------------------------------------------------------------------------------------
    // Update.
    // --------------------------------------------------------------------------------------------

    // The filmic curve ALU only tonemapper from John Hable's presentation.
    RGB tonemap_filmic_ALU(RGB color) {
        color = saturate(color - 0.004f);
        color = (color * (6.2f * color + 0.5f)) / (color * (6.2f * color + 1.7f) + 0.06f);

        // result has 1/2.2 baked in
        return gammacorrect(color, 2.2f);
    }

    void color_grade_image(Image image, RGB* output) {
        float l_scale = luminance_scale();
        int width = image.get_width(), height = image.get_height();

        static Image bloom_image = Images::UID::invalid_UID();
        if (m_bloom.enabled) { // Bloom
            static Image high_intensity_image = Images::UID::invalid_UID();
            if (!high_intensity_image.exists())
                high_intensity_image = Images::create2D("high intensity", PixelFormat::RGB_Float, 1.0f, Vector2ui(width, height));

            #pragma omp parallel for schedule(dynamic, 16)
            for (int i = 0; i < (int)image.get_pixel_count(); ++i) {
                RGBA pixel = image.get_pixel(i);
                RGBA high_intensity_pixel = RGBA(max(pixel.r - m_bloom.threshold, 0.0f),
                                                 max(pixel.g - m_bloom.threshold, 0.0f),
                                                 max(pixel.b - m_bloom.threshold, 0.0f), pixel.a);
                high_intensity_image.set_pixel(high_intensity_pixel, i);
            }

            if (!bloom_image.exists())
                bloom_image = Images::create2D("high intensity", PixelFormat::RGB_Float, 1.0f, Vector2ui(width, height));
            float pixel_support = image.get_height() * max(0.001f, m_bloom.support);
            float std_dev = pixel_support / 4.0f;
            Blur::gaussian(high_intensity_image.get_ID(), std_dev, bloom_image.get_ID());
        }

        { // Tonemap
            float bloom_threshold = m_bloom.enabled ? m_bloom.threshold : INFINITY;
            #pragma omp parallel for schedule(dynamic, 16)
            for (int i = 0; i < (int)image.get_pixel_count(); ++i) {
                RGB pixel = image.get_pixel(i).rgb();
                RGB bloom = m_bloom.enabled ? bloom_image.get_pixel(i).rgb() : RGB::black();
                RGB adjusted_color = (RGB(min(pixel.r, bloom_threshold), min(pixel.g, bloom_threshold), min(pixel.b, bloom_threshold)) + bloom) * l_scale;
                if (m_operator == Operator::Reinhard)
                    adjusted_color = CameraEffects::reinhard(adjusted_color, m_reinhard_whitepoint * m_reinhard_whitepoint);
                else if (m_operator == Operator::FilmicAlu)
                    adjusted_color = tonemap_filmic_ALU(adjusted_color);
                else if (m_operator == Operator::Uncharted2)
                    adjusted_color = CameraEffects::uncharted2(adjusted_color, m_uncharted2);
                else if (m_operator == Operator::Unreal4)
                    adjusted_color = CameraEffects::unreal4(adjusted_color, m_unreal4);
                output[i] = gammacorrect(adjusted_color, 1.0f / 2.2f);
            }
        }
    }

    void update(Engine& engine) {

        if (m_upload_image) {

            color_grade_image(m_input, m_tonemapped_pixels);

            glBindTexture(GL_TEXTURE_2D, m_tex_ID);
            int width = m_input.get_width(), height = m_input.get_height();
            const GLint BASE_IMAGE_LEVEL = 0;
            const GLint NO_BORDER = 0;
            glTexImage2D(GL_TEXTURE_2D, BASE_IMAGE_LEVEL, GL_RGB, width, height, NO_BORDER, GL_RGB, GL_FLOAT, m_tonemapped_pixels);

            m_upload_image = false;
        }

        render_image(engine.get_window(), m_tex_ID, m_input.get_width(), m_input.get_height());

        if (m_histogram.visualize) {

            fill_histogram_texture(m_histogram.histogram, m_histogram.texture_pixels);

            glBindTexture(GL_TEXTURE_2D, m_histogram.texture_ID);
            const GLint BASE_IMAGE_LEVEL = 0;
            const GLint NO_BORDER = 0;
            glTexImage2D(GL_TEXTURE_2D, BASE_IMAGE_LEVEL, GL_RGBA, m_histogram.size, m_histogram.size, NO_BORDER, GL_RGBA, GL_UNSIGNED_BYTE, m_histogram.texture_pixels);

            glBegin(GL_QUADS); {

                glTexCoord2f(0.0f, 0.0f);
                glVertex3f(0.3f, 0.3f, 0.f);

                glTexCoord2f(0.0f, 1.0f);
                glVertex3f(0.9f, 0.3f, 0.f);

                glTexCoord2f(1.0f, 1.0f);
                glVertex3f(0.9f, 0.9f, 0.f);

                glTexCoord2f(1.0f, 0.0f);
                glVertex3f(0.3f, 0.9f, 0.f);

            } glEnd();
        }

        AntTweakBar::handle_input(engine);
        TwDraw();
    }

    static void update(Engine& engine, void* color_grader) {
        ((ColorGrader::Implementation*)color_grader)->update(engine);
    }
};

ColorGrader::ColorGrader(std::vector<char*> args, Cogwheel::Core::Engine& engine) {
    m_impl = new Implementation(args, engine);
}
