// Komodo image blurer.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <Blurer.h>
#include <Utils.h>

#include <AntTweakBar/AntTweakBar.h>
#include <Cogwheel/Core/Engine.h>
#include <ImageOperations/Blur.h>

using namespace Cogwheel::Assets;
using namespace Cogwheel::Core;
using namespace Cogwheel::Math;

struct Blurer::Implementation final {
    // --------------------------------------------------------------------------------------------
    // Members.
    // --------------------------------------------------------------------------------------------
    float m_std_dev;

    Image m_input;
    Image m_blurred_image; // RGB with gamma 1, to allows us to easily map the data.
    std::string m_output_path = "";

    GLuint m_tex_ID = 0u;
    bool m_update_image = true;

    TwBar* m_gui;

    // --------------------------------------------------------------------------------------------
    // Print help message.
    // --------------------------------------------------------------------------------------------
    static void print_usage() {
        static char* usage =
            "usage Komodo Blurring:\n"
            "  -h | --help: Show command line usage for Komodo Blurring.\n"
            "     | --input <path>: Path to the image to be tonemapped.\n"
            "     | --output <path>: Path to where to store the final image.\n"
            "  -s | --std-dev <value>: Standard deviation for gaussian blur.\n";

        printf("%s", usage);
    }

    // --------------------------------------------------------------------------------------------
    // Parse command line arguments.
    // --------------------------------------------------------------------------------------------
    std::string parse_arguments(std::vector<char*> args) {
        std::string input_path;
        for (int i = 0; i < args.size(); ++i) {
            std::string arg = args[i];
            if (arg.compare("--input") == 0)
                input_path = args[++i];
            else if (arg.compare("--output") == 0)
                m_output_path = args[++i];
            else if (arg.compare("--std-dev") == 0)
                m_std_dev = float(atof(args[++i]));
            else
                printf("Unknown argument: %s\n", args[i]);
        }

        return input_path;
    }

    // --------------------------------------------------------------------------------------------
    // Constructor.
    // --------------------------------------------------------------------------------------------
    Implementation(std::vector<char*> args, Engine& engine) {
        m_std_dev = 4;

        // Parse arguments
        std::string input_path = parse_arguments(args);

        // Load input image
        m_input = load_image(input_path);
        if (!m_input.exists()) {
            printf("  error: Could not load image at '%s'\n", input_path.c_str());
            m_input = create_error_image();
        }
        engine.get_window().set_name("Komodo - " + m_input.get_name());

        Vector2ui size = Vector2ui(m_input.get_width(), m_input.get_height());
        m_blurred_image = Images::create2D("blurred_" + m_input.get_name(), PixelFormat::RGB_Float, 2.2f, size);

        bool headless = engine.get_window().get_width() == 0 && engine.get_window().get_height() == 0;
        if (headless) {
            if (m_output_path.size() != 0) {
                // Blur and store the image
                Vector2ui size = { m_input.get_width(), m_input.get_height() };
                Image output_image = blur_image(m_input);
                store_image(output_image, m_output_path);
            }
        }
        else {

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

            m_gui = setup_gui();
            engine.add_mutating_callback([&]{ this->update(engine); });
        }
    }

    // --------------------------------------------------------------------------------------------
    // GUI.
    // --------------------------------------------------------------------------------------------
#undef WRAP_ANT_PROPERTY
#define WRAP_ANT_PROPERTY(member_name, T) \
[](const void* input_data, void* client_data) { \
    Blurer::Implementation* data = (Blurer::Implementation*)client_data; \
    data->member_name = *(T*)input_data; \
    data->m_update_image = true; \
}, \
[](void* value, void* client_data) { \
    *(T*)value = ((Blurer::Implementation*)client_data)->member_name; \
}

    TwBar* setup_gui() {
        TwInit(TW_OPENGL, nullptr);
        TwDefine("TW_HELP visible=false");  // Ant help bar is hidden.

        TwBar* bar = TwNewBar("Blurer");

        TwAddVarCB(bar, "Standard deviation", TW_TYPE_FLOAT, WRAP_ANT_PROPERTY(m_std_dev, float), this, "step=0.1 group=Guassian");

        // Save button.
        if (m_output_path.size() != 0) {
            auto save_image = [](void* client_data) {
                // Tonemap and store the image
                Implementation* data = (Implementation*)client_data;
                Vector2ui size = { data->m_input.get_width(), data->m_input.get_height() };
                store_image(data->m_blurred_image, data->m_output_path);
            };
            TwAddButton(bar, "Save", save_image, this, "");
        }


        return bar;
    }

    // --------------------------------------------------------------------------------------------
    // Update.
    // --------------------------------------------------------------------------------------------
    Image blur_image(Image image) {
        ImageOperations::Blur::gaussian(image.get_ID(), m_std_dev, m_blurred_image.get_ID());
        return m_blurred_image;
    }

    void update(Engine& engine) {

        if (m_update_image) {

            Image blurred_image = blur_image(m_input);

            glBindTexture(GL_TEXTURE_2D, m_tex_ID);
            int width = m_input.get_width(), height = m_input.get_height();
            const GLint BASE_IMAGE_LEVEL = 0;
            const GLint NO_BORDER = 0;
            glTexImage2D(GL_TEXTURE_2D, BASE_IMAGE_LEVEL, GL_RGB, width, height, NO_BORDER, GL_RGB, GL_FLOAT, blurred_image.get_pixels<RGB>());

            m_update_image = false;
        }

        render_image(engine.get_window(), m_tex_ID, m_input.get_width(), m_input.get_height());

        AntTweakBar::handle_input(engine);
        TwDraw();
    }

    static void update(Engine& engine, void* blurer) {
        ((Blurer::Implementation*)blurer)->update(engine);
    }
};

Blurer::Blurer(std::vector<char*> args, Engine& engine) {
    m_impl = new Implementation(args, engine);
}
