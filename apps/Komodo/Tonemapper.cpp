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

using namespace Cogwheel::Assets;
using namespace Cogwheel::Core;
using namespace Cogwheel::Math;

struct Tonemapper::Implementation final {

    // --------------------------------------------------------------------------------------------
    // Members
    // --------------------------------------------------------------------------------------------
    enum class Operator { Linear, Reinhard, Uncharted2 };
    Operator m_operator = Operator::Uncharted2;
    Image m_input;
    std::string m_output_path;

    GLuint m_tex_ID = 0u;
    bool m_upload_image = true;

    // Exposure members
    float m_exposure_bias = 0.0f;
    
    // Tonemapping members
    float m_reinhard_whitepoint = 1.0f;

    float m_uncharted2_shoulder_strength = 0.15f;
    float m_uncharted2_linear_strength = 0.5f;
    float m_uncharted2_linear_angle = 0.1f;
    float m_uncharted2_toe_strength = 0.2f;
    float m_uncharted2_toe_numerator = 0.02f;
    float m_uncharted2_toe_denominator = 0.3f;
    float m_uncharted2_linear_white = 11.2f;

    TwBar* m_gui = nullptr;

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
    *static_cast<T*>(value) = static_cast<Tonemapper::Implementation*>(client_data)->member_name; \
}

    // --------------------------------------------------------------------------------------------
    // GUI.
    // --------------------------------------------------------------------------------------------
    TwBar* setup_gui() {
        TwInit(TW_OPENGL, nullptr);
        TwDefine("TW_HELP visible=false");  // Ant help bar is hidden.

        TwBar* bar = TwNewBar("Tonemapper");

        { // Exposure mapping

            TwAddVarCB(bar, "Exposure bias", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_exposure_bias, float), this, "step=0.1 group='Exposure'");

            auto auto_exposure = [](void* client_data) {
                Tonemapper::Implementation* data = (Tonemapper::Implementation*)client_data;
                float average_log_luminance = ImageOperations::Exposure::average_log_luminance(data->m_input.get_ID());
                data->m_exposure_bias = exp2(average_log_luminance);
                data->m_upload_image = true;
            };
            TwAddButton(bar, "Auto adjust", auto_exposure, this, "group=Exposure");
        }

        { // Tonemapping
            TwEnumVal operators[] = { { int(Operator::Linear), "Linear" },
                                      { int(Operator::Reinhard), "Reinhard" }, 
                                      { int(Operator::Uncharted2), "Uncharted2" } };
            TwType AntOperatorEnum = TwDefineEnum("Operators", operators, 3);

            auto set_m_operator = [](const void* input_data, void* client_data) {
                Tonemapper::Implementation* data = (Tonemapper::Implementation*)client_data;
                data->m_operator = *(Operator*)input_data;
                data->m_upload_image = true;

                auto show_reinhard = std::string("Tonemapper/Reinhard visible=") + (data->m_operator == Operator::Reinhard ? "true" : "false");
                TwDefine(show_reinhard.c_str());

                auto show_uncharted2 = std::string("Tonemapper/Uncharted2 visible=") + (data->m_operator == Operator::Uncharted2 ? "true" : "false");
                TwDefine(show_uncharted2.c_str());
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
                TwAddVarCB(bar, "Shoulder strength", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_uncharted2_shoulder_strength, float), this, "min=0 step=0.1 group=Uncharted2");
                TwAddVarCB(bar, "Linear strength", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_uncharted2_linear_strength, float), this, "min=0 step=0.1 group=Uncharted2");
                TwAddVarCB(bar, "Linear angle", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_uncharted2_linear_angle, float), this, "min=0 step=0.1 group=Uncharted2");
                TwAddVarCB(bar, "Toe strength", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_uncharted2_toe_strength, float), this, "min=0 step=0.1 group=Uncharted2");
                TwAddVarCB(bar, "Toe numerator", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_uncharted2_toe_numerator, float), this, "min=0 step=0.1 group=Uncharted2");
                TwAddVarCB(bar, "Toe denominator", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_uncharted2_toe_denominator, float), this, "min=0 step=0.1 group=Uncharted2");
                TwAddVarCB(bar, "Linear white", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_uncharted2_linear_white, float), this, "min=0 step=0.1 group=Uncharted2");

                TwDefine("Tonemapper/Uncharted2 group='Tonemapping' label='Uncharted 2'");
            }

            auto tonemapper = Operator::Uncharted2;
            set_m_operator(&tonemapper, this);
        }

        return bar;
    }

    // --------------------------------------------------------------------------------------------
    // Constructor.
    // --------------------------------------------------------------------------------------------
    Implementation(std::vector<char*> args, Cogwheel::Core::Engine& engine) {

        // Create GL texture.
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &m_tex_ID);
        glBindTexture(GL_TEXTURE_2D, m_tex_ID);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

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
    void update(Engine& engine) {

        if (m_upload_image) {

            float exposure = exp2f(m_exposure_bias);
            int width = m_input.get_width(), height = m_input.get_height();
            RGB* gamma_corrected_pixels = new RGB[m_input.get_pixel_count()];
            #pragma omp parallel for schedule(dynamic, 16)
            for (int i = 0; i < (int)m_input.get_pixel_count(); ++i) {
                int x = i % width, y = i / width;
                RGB adjusted_color = m_input.get_pixel(Vector2ui(x, y)).rgb() * exposure;
                if (m_operator == Operator::Reinhard)
                    adjusted_color = Tonemapping::reinhard(adjusted_color, m_reinhard_whitepoint * m_reinhard_whitepoint);
                else if (m_operator == Operator::Uncharted2)
                    adjusted_color = Tonemapping::uncharted2(adjusted_color, m_uncharted2_shoulder_strength, m_uncharted2_linear_strength, m_uncharted2_linear_angle, m_uncharted2_toe_strength, m_uncharted2_toe_numerator, m_uncharted2_toe_denominator, m_uncharted2_linear_white);
                gamma_corrected_pixels[i] = gammacorrect(adjusted_color, 1.0f / 2.2f);
            }

            glBindTexture(GL_TEXTURE_2D, m_tex_ID);
            const GLint BASE_IMAGE_LEVEL = 0;
            const GLint NO_BORDER = 0;
            glTexImage2D(GL_TEXTURE_2D, BASE_IMAGE_LEVEL, GL_RGB, width, height, NO_BORDER, GL_RGB, GL_FLOAT, gamma_corrected_pixels);

            m_upload_image = false;
        }

        render_image(engine.get_window(), m_tex_ID);

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
