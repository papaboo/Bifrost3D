// Komodo Image Tool.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <Diff.h>

#include <Cogwheel/Core/Engine.h>
#include <Cogwheel/Input/Keyboard.h>

#include <GLFWDriver.h>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <GL/gl.h>
#undef RGB

#include <ImageOperations/Diff.h>

#include <omp.h>

using namespace Cogwheel::Assets;
using namespace Cogwheel::Core;
using namespace Cogwheel::Input;
using namespace Cogwheel::Math;

// Global state
std::vector<char*> g_args;
std::vector<Cogwheel::Assets::Image> g_images;

void update(Engine& engine, void* none) {
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

    static int image_index = 0;
    static int uploaded_image_index = -1;
    if (keyboard->was_released(Keyboard::Key::Left))
        --image_index;
    if (keyboard->was_released(Keyboard::Key::Right))
        ++image_index;
    image_index = clamp(image_index, 0, int(g_images.size() - 1));

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
            Image image = g_images[image_index];
            int width = image.get_width(), height = image.get_height();
            RGB* gamma_corrected_pixels = new RGB[image.get_pixel_count()];
            // #pragma omp parallel for schedule(dynamic, 16)
            for (int i = 0; i < (int)image.get_pixel_count(); ++i) {
                int x = i % width, y = i / width;
                gamma_corrected_pixels[i] = gammacorrect(image.get_pixel(Vector2ui(x, y)).rgb(), 1.0f / 2.2f);
            }
            glTexImage2D(GL_TEXTURE_2D, BASE_IMAGE_LEVEL, GL_RGB, width, height, NO_BORDER, GL_RGB, GL_FLOAT, gamma_corrected_pixels);
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
    engine.get_window().set_name("Komodo");

    Images::allocate(3u);


    std::string operation_name = g_args[0];
    g_args.erase(g_args.begin());

    if (std::string(operation_name).compare("--diff") == 0)
        g_images = Diff::apply(g_args);

    if (g_images.empty()) {
        // Create a default red and white image.
        Image error_img = Images::create2D("No images loaded", PixelFormat::RGBA32, 2.2f, Vector2ui(16,16));
        unsigned char* pixels = (unsigned char*)error_img.get_pixels();
        for (unsigned int y = 0; y < error_img.get_height(); ++y) {
            for (unsigned int x = 0; x < error_img.get_width(); ++x) {
                unsigned char* pixel = pixels + (x + y * error_img.get_width()) * 4u;
                unsigned char intensity = ((x & 1) == (y & 1)) ? 2 : 255;
                pixel[0] = 255u;
                pixel[1] = pixel[2] = intensity;
                pixel[3] = 255u;
            }
        }
        g_images.push_back(error_img);
    }

    engine.add_mutating_callback(update, nullptr);

    return 0;
}

void print_usage() {
    printf("TODO\n");
}

int main(int argc, char** argv) {
    printf("Komodo Image Tool\n");

    if (argc == 1 || std::string(argv[1]).compare("-h") == 0 || std::string(argv[1]).compare("--help") == 0) {
        print_usage();
        return 0;
    }

    bool headless = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]).compare("-l") == 0 || std::string(argv[i]).compare("--headless") == 0)
            headless = true;
        else
            g_args.push_back(argv[i]);

    if (headless) {
        Engine* engine = new Engine("Headless");
        initialize(*engine);
    } else
        GLFWDriver::run(initialize, nullptr);
}
