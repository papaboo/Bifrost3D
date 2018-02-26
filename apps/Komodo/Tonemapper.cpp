// Komodo image tonemapper.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <Tonemapper.h>
#include <Utils.h>

#include <Cogwheel/Core/Engine.h>
#include <Cogwheel/Math/Tonemapping.h>
#include <ImageOperations/Exposure.h>

#include <AntTweakBar/AntTweakBar.h>

#include <array>

using namespace Cogwheel::Assets;
using namespace Cogwheel::Core;
using namespace Cogwheel::Math;

using Vector4uc = Vector4<unsigned char>;

struct Tonemapper::Implementation final {

    // --------------------------------------------------------------------------------------------
    // Members
    // --------------------------------------------------------------------------------------------
    enum class Operator { Linear, Reinhard, FilmicAlu, Uncharted2, Unreal4 };
    Operator m_operator = Operator::Unreal4;
    Image m_input;
    std::string m_output_path;

    GLuint m_tex_ID = 0u;
    bool m_upload_image = true;

    // Exposure members and helpers
    float m_exposure_bias = 0.0f;
    float luminance_scale() { return exp2(m_exposure_bias); }
    float scene_key() { return 0.5f / luminance_scale(); }

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
    float m_reinhard_whitepoint = 1.0f;

    struct {
        float shoulder_strength = 0.22f;
        float linear_strength = 0.3f;
        float linear_angle = 0.1f;
        float toe_strength = 0.2f;
        float toe_numerator = 0.01f;
        float toe_denominator = 0.3f;
        float linear_white = 11.2f;
    } m_uncharted2;

    struct {
        float slope = 0.91f;
        float toe = 0.53f;
        float shoulder = 0.23f;
        float black_clip = 0.0f;
        float white_clip = 0.035f;
        float desaturate = 1.0f;
    } m_unreal4;

    TwBar* m_gui = nullptr;

    // --------------------------------------------------------------------------------------------
    // Histogram helpers
    // --------------------------------------------------------------------------------------------

    void compute_histogram_high_low_normalized_index(float& low_normalized_index, float& high_normalized_index) {
        // Compute log luminance values from histogram.
        int min_pixel_count = int(m_input.get_pixel_count() * m_histogram.min_percentage);
        int max_pixel_count = int(m_input.get_pixel_count() * m_histogram.max_percentage);

        int previous_pixel_count = 0;
        for (int i = 0; i < m_histogram.histogram.size(); ++i) {
            int current_pixel_count = previous_pixel_count + m_histogram.histogram[i];
            // TODO check edge cases.
            if (previous_pixel_count <= min_pixel_count && min_pixel_count < current_pixel_count)
                low_normalized_index = i / float(m_histogram.histogram.size()); // TODO base on relative offset between previous and current.
            if (previous_pixel_count <= max_pixel_count && max_pixel_count < current_pixel_count)
                high_normalized_index = i / float(m_histogram.histogram.size()); // TODO base on relative offset between previous and current.

            previous_pixel_count = current_pixel_count;
        }
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
            "     | --reinhard: Apply reinhard tonemapper.\n"
            "     | --filmic: Apply filmic tonemapper.\n"
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
    Tonemapper::Implementation* data = (Tonemapper::Implementation*)client_data; \
    data->member_name = *(T*)input_data; \
    data->m_upload_image = true; \
}, \
[](void* value, void* client_data) { \
    *(T*)value = ((Tonemapper::Implementation*)client_data)->member_name; \
}

    // --------------------------------------------------------------------------------------------
    // GUI.
    // --------------------------------------------------------------------------------------------
    TwBar* setup_gui() {
        TwInit(TW_OPENGL, nullptr);
        TwDefine("TW_HELP visible=false");  // Ant help bar is hidden.

        TwBar* bar = TwNewBar("Tonemapper");

        { // Exposure mapping

            TwAddVarCB(bar, "Bias", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_exposure_bias, float), this, "step=0.1 group=Exposure");
            TwAddVarCB(bar, "Scene key", TW_TYPE_FLOAT, nullptr, [](void* value, void* client_data) {
                *(float*)value = ((Tonemapper::Implementation*)client_data)->scene_key();
            }, this, "step=0.001 group=Exposure");
            TwAddVarCB(bar, "Luminance scale", TW_TYPE_FLOAT, nullptr, [](void* value, void* client_data) {
                *(float*)value = ((Tonemapper::Implementation*)client_data)->luminance_scale();
            }, this, "step=0.001 group=Exposure");

            auto reinhard_auto_exposure = [](void* client_data) {
                Tonemapper::Implementation* data = (Tonemapper::Implementation*)client_data;
                float scene_key = ImageOperations::Exposure::log_average_luminance(data->m_input.get_ID());
                float linear_exposure = 0.5f / scene_key;
                data->m_exposure_bias = log2(linear_exposure);
                data->m_upload_image = true;
            };
            TwAddButton(bar, "Set from log-average", reinhard_auto_exposure, this, "group=Exposure");

            auto auto_geometric_mean = [](void* client_data) {
                Tonemapper::Implementation* data = (Tonemapper::Implementation*)client_data;
                float log_average_luminance = ImageOperations::Exposure::log_average_luminance(data->m_input.get_ID());
                log_average_luminance = max(log_average_luminance, 0.001f);
                float key_value = 1.03f - (2.0f / (2 + log2(log_average_luminance + 1)));
                float linear_exposure = (key_value / log_average_luminance);
                data->m_exposure_bias = log2(max(linear_exposure, 0.0001f));

                data->m_upload_image = true;
            };
            TwAddButton(bar, "Set from geometric mean", auto_geometric_mean, this, "group=Exposure");

            { // Histogram
                m_histogram.texture_pixels = new Vector4uc[m_histogram.size * m_histogram.size];

                auto histogram_auto_exposure = [](void* client_data) {
                    Tonemapper::Implementation* data = (Tonemapper::Implementation*)client_data;

                    std::fill(data->m_histogram.histogram.begin(), data->m_histogram.histogram.end(), 0u);
                    ImageOperations::Exposure::log_luminance_histogram(data->m_input.get_ID(), data->m_histogram.min_log_luminance, data->m_histogram.max_log_luminance, 
                                                                       data->m_histogram.histogram.begin(), data->m_histogram.histogram.end());

                    // Compute log luminance values from histogram.
                    int min_pixel_count = int(data->m_input.get_pixel_count() * data->m_histogram.min_percentage);
                    int max_pixel_count = int(data->m_input.get_pixel_count() * data->m_histogram.max_percentage);

                    float low_normalized_index, high_normalized_index;
                    data->compute_histogram_high_low_normalized_index(low_normalized_index, high_normalized_index);
                    float low_log_luminance = lerp(data->m_histogram.min_log_luminance, data->m_histogram.max_log_luminance, low_normalized_index);
                    float high_log_luminance = lerp(data->m_histogram.min_log_luminance, data->m_histogram.max_log_luminance, high_normalized_index);

                    // TODO Check how Unreal sets this.
                    float target_log_luminance = (low_log_luminance + high_log_luminance) * 0.5f;
                    float linear_exposure = 0.5f / exp2(target_log_luminance);
                    data->m_exposure_bias = log2(linear_exposure);

                    data->m_upload_image = true;
                };
                TwAddButton(bar, "Set from histogram", histogram_auto_exposure, this, "group=Histogram");

                TwAddVarCB(bar, "Min percentage", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_histogram.min_percentage, float), this, "step=0.05 group=Histogram");
                TwAddVarCB(bar, "Max percentage", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_histogram.max_percentage, float), this, "step=0.05 group=Histogram");
                TwAddVarCB(bar, "Min log luminance", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_histogram.min_log_luminance, float), this, "step=0.1 group=Histogram");
                TwAddVarCB(bar, "Max log luminance", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_histogram.max_log_luminance, float), this, "step=0.1 group=Histogram");
                TwAddVarCB(bar, "Visualize", TW_TYPE_BOOLCPP, WRAP_ANT_PROPERTY(m_histogram.visualize, bool), this, "group=Histogram");

                TwDefine("Tonemapper/Histogram group='Exposure'");
            }
        }

        { // Tonemapping
            TwEnumVal operators[] = { { int(Operator::Linear), "Linear" },
                                      { int(Operator::Reinhard), "Reinhard" }, 
                                      { int(Operator::FilmicAlu), "FilmicAlu" },
                                      { int(Operator::Uncharted2), "Uncharted2" },
                                      { int(Operator::Unreal4), "Unreal4" } };
            TwType AntOperatorEnum = TwDefineEnum("Operators", operators, 5);

            auto set_m_operator = [](const void* input_data, void* client_data) {
                Tonemapper::Implementation* data = (Tonemapper::Implementation*)client_data;
                data->m_operator = *(Operator*)input_data;
                data->m_upload_image = true;

                auto show_reinhard = std::string("Tonemapper/Reinhard visible=") + (data->m_operator == Operator::Reinhard ? "true" : "false");
                TwDefine(show_reinhard.c_str());

                auto show_uncharted2 = std::string("Tonemapper/Uncharted2 visible=") + (data->m_operator == Operator::Uncharted2 ? "true" : "false");
                TwDefine(show_uncharted2.c_str());

                auto show_unreal4 = std::string("Tonemapper/Unreal4 visible=") + (data->m_operator == Operator::Unreal4 ? "true" : "false");
                TwDefine(show_unreal4.c_str());
            };
            auto get_m_operator = [](void* value, void* client_data) {
                *(Operator*)value = ((Tonemapper::Implementation*)client_data)->m_operator;
            };
            TwAddVarCB(bar, "Operator", AntOperatorEnum, set_m_operator, get_m_operator, this, "group='Tonemapping'");

            { // Reinhard
                TwAddVarCB(bar, "White point", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_reinhard_whitepoint, float), this, "min=0 step=0.1 group='Reinhard'");

                TwDefine("Tonemapper/Reinhard group='Tonemapping'");
            }

            { // Uncharted 2 filmic
                TwAddVarCB(bar, "Shoulder strength", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_uncharted2.shoulder_strength, float), this, "min=0 step=0.1 group=Uncharted2");
                TwAddVarCB(bar, "Linear strength", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_uncharted2.linear_strength, float), this, "min=0 step=0.1 group=Uncharted2");
                TwAddVarCB(bar, "Linear angle", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_uncharted2.linear_angle, float), this, "min=0 step=0.1 group=Uncharted2");
                TwAddVarCB(bar, "Toe strength", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_uncharted2.toe_strength, float), this, "min=0 step=0.1 group=Uncharted2");
                TwAddVarCB(bar, "Toe numerator", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_uncharted2.toe_numerator, float), this, "min=0 step=0.1 group=Uncharted2");
                TwAddVarCB(bar, "Toe denominator", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_uncharted2.toe_denominator, float), this, "min=0 step=0.1 group=Uncharted2");
                TwAddVarCB(bar, "Linear white", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_uncharted2.linear_white, float), this, "min=0 step=0.1 group=Uncharted2");

                TwDefine("Tonemapper/Uncharted2 group='Tonemapping' label='Uncharted 2'");
            }

            { // Unreal 4 filmic
                { // Presets
                    enum class Presets { None, Default, Uncharted2, HP, ACES, Legacy };
                    TwEnumVal ant_presets[] = { { int(Presets::None), "Select preset" },
                                                { int(Presets::Default), "Default" },
                                                { int(Presets::Uncharted2), "Uncharted2" },
                                                { int(Presets::HP), "HP" },
                                                { int(Presets::ACES), "ACES" },
                                                { int(Presets::Legacy), "Legacy" } };
                    TwType AntPresetsEnum = TwDefineEnum("Presets", ant_presets, 6);

                    auto set_preset = [](const void* input_data, void* client_data) {
                        Tonemapper::Implementation* data = (Tonemapper::Implementation*)client_data;
                        Presets preset = *(Presets*)input_data;

                        switch (preset) {
                        case Presets::None:
                            // Do nothing
                            break;
                        case Presets::Uncharted2:
                            data->m_unreal4.slope = 0.63f;
                            data->m_unreal4.toe = 0.55f;
                            data->m_unreal4.shoulder = 0.47f;
                            data->m_unreal4.black_clip = 0.0f;
                            data->m_unreal4.white_clip = 0.01f;
                            data->m_unreal4.desaturate = 0.0f;
                            break;
                        case Presets::HP:
                            data->m_unreal4.slope = 0.65f;
                            data->m_unreal4.toe = 0.63f;
                            data->m_unreal4.shoulder = 0.45f;
                            data->m_unreal4.black_clip = 0.0f;
                            data->m_unreal4.white_clip = 0.0f;
                            data->m_unreal4.desaturate = 1.0f;
                            break;
                        case Presets::ACES:
                            data->m_unreal4.slope = 0.91f;
                            data->m_unreal4.toe = 0.53f;
                            data->m_unreal4.shoulder = 0.23f;
                            data->m_unreal4.black_clip = 0.0f;
                            data->m_unreal4.white_clip = 0.035f;
                            data->m_unreal4.desaturate = 1.0f;
                            break;
                        case Presets::Legacy:
                            data->m_unreal4.slope = 0.98f;
                            data->m_unreal4.toe = 0.3f;
                            data->m_unreal4.shoulder = 0.22f;
                            data->m_unreal4.black_clip = 0.0f;
                            data->m_unreal4.white_clip = 0.025f;
                            data->m_unreal4.desaturate = 1.0f;
                            break;
                        case Presets::Default:
                        default:
                            data->m_unreal4.slope = 0.91f;
                            data->m_unreal4.toe = 0.53f;
                            data->m_unreal4.shoulder = 0.23f;
                            data->m_unreal4.black_clip = 0.0f;
                            data->m_unreal4.white_clip = 0.035f;
                            data->m_unreal4.desaturate = 1.0f;
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
                TwAddVarCB(bar, "Desaturate", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_unreal4.desaturate, float), this, "step=0.1 group=Unreal4");

                TwDefine("Tonemapper/Unreal4 group='Tonemapping'");
            }

            auto tonemapper = Operator::Unreal4;
            set_m_operator(&tonemapper, this);
        }

        return bar;
    }

    // --------------------------------------------------------------------------------------------
    // Constructor.
    // --------------------------------------------------------------------------------------------
    Implementation(std::vector<char*> args, Cogwheel::Core::Engine& engine) {

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
        
        if (args.size() == 0 || std::string(args[0]).compare("-h") == 0 || std::string(args[0]).compare("--help") == 0)
            print_usage();
        else {

            // Parse arguments
            std::string input_path = parse_arguments(args);

            // Load input image
            m_input = load_image(input_path);
            if (!m_input.exists()) {
                printf("  error: Could not load image at '%s'\n", input_path.c_str());
                m_input = create_error_image();
            }
            engine.get_window().set_name("Komodo - " + m_input.get_name());

            m_gui = setup_gui();
        }

        engine.add_mutating_callback(Tonemapper::Implementation::update, this);
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

    void update(Engine& engine) {

        if (m_upload_image) {
            float l_scale = luminance_scale();
            int width = m_input.get_width(), height = m_input.get_height();
            RGB* gamma_corrected_pixels = new RGB[m_input.get_pixel_count()]; // TODO Preallocate
            #pragma omp parallel for schedule(dynamic, 16)
            for (int i = 0; i < (int)m_input.get_pixel_count(); ++i) {
                int x = i % width, y = i / width;
                RGB adjusted_color = m_input.get_pixel(Vector2ui(x, y)).rgb() * l_scale;
                if (m_operator == Operator::Reinhard)
                    adjusted_color = Tonemapping::reinhard(adjusted_color, m_reinhard_whitepoint * m_reinhard_whitepoint);
                else if (m_operator == Operator::FilmicAlu)
                    adjusted_color = tonemap_filmic_ALU(adjusted_color);
                else if (m_operator == Operator::Uncharted2)
                    adjusted_color = Tonemapping::uncharted2(adjusted_color, m_uncharted2.shoulder_strength, m_uncharted2.linear_strength, m_uncharted2.linear_angle, m_uncharted2.toe_strength, m_uncharted2.toe_numerator, m_uncharted2.toe_denominator, m_uncharted2.linear_white);
                else if (m_operator == Operator::Unreal4)
                    adjusted_color = Tonemapping::unreal4(adjusted_color, m_unreal4.slope, m_unreal4.toe, m_unreal4.shoulder, m_unreal4.black_clip, m_unreal4.white_clip, m_unreal4.desaturate);
                gamma_corrected_pixels[i] = gammacorrect(adjusted_color, 1.0f / 2.2f);
            }

            glBindTexture(GL_TEXTURE_2D, m_tex_ID);
            const GLint BASE_IMAGE_LEVEL = 0;
            const GLint NO_BORDER = 0;
            glTexImage2D(GL_TEXTURE_2D, BASE_IMAGE_LEVEL, GL_RGB, width, height, NO_BORDER, GL_RGB, GL_FLOAT, gamma_corrected_pixels);

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

    static void update(Engine& engine, void* tonemapper) {
        ((Tonemapper::Implementation*)tonemapper)->update(engine);
    }
};

Tonemapper::Tonemapper(std::vector<char*> args, Cogwheel::Core::Engine& engine) {
    m_impl = new Implementation(args, engine);
}
