// Komodo image comparison.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _KOMODO_COMPARER_H_
#define _KOMODO_COMPARER_H_

#include <Utils.h>

#include <Cogwheel/Input/Keyboard.h>
#include <ImageOperations/Compare.h>

#include <vector>

#include <GL/gl.h>

using namespace Cogwheel::Assets;

class Comparer final {
public:

    enum class Algorithm { SSIM, RMS };

    static void print_usage() {
        char* usage =
            "usage Komodo Image Comparison:\n"
            "  -h | --help: Show command line usage for Komodo Image Comparison.\n"
            "     | --ssim: Compare using Structured Similiarity Index.\n"
            "     | --rms: Compare using root mean square.\n"
            "     | --reference <path>: Path to the reference image.\n"
            "     | --target <path>: Path to the target image.\n"
            "     | --diff <path>: Path to store the diff image to.\n";

        printf("%s", usage);
    }

    static std::vector<Image> apply(std::vector<char*> args) {

        if (args.size() == 0 || std::string(args[0]).compare("-h") == 0 || std::string(args[0]).compare("--help") == 0) {
            print_usage();
            return std::vector<Image>();
        }

        Algorithm algorithm = Algorithm::SSIM;
        std::string reference_path;
        std::string target_path;
        std::string diff_path;

        for (int i = 0; i < args.size(); ++i) {
            std::string arg = args[i];
            if (arg.compare("--reference") == 0)
                reference_path = args[++i];
            else if (arg.compare("--target") == 0)
                target_path = args[++i];
            else if (arg.compare("--diff") == 0)
                diff_path = args[++i];
            else if (arg.compare("--ssim") == 0)
                algorithm = Algorithm::SSIM;
            else if (arg.compare("--rms") == 0)
                algorithm = Algorithm::RMS;
            else
                printf("Unknown argument: %s\n", args[i]);
        }

        std::vector<Image> images;

        Image reference = load_image(reference_path);
        if (!reference.exists()) {
            printf("  error: Could not load reference image '%s'\n", reference_path.c_str());
            return std::vector<Image>();
        } else
            images.push_back(reference);

        Image target = load_image(target_path);
        if (!target.exists()) {
            printf("  error: Could not load target image '%s'\n", target_path.c_str());
            return std::vector<Image>();
        } else
            images.push_back(target);

        Image diff_image = Images::create2D(diff_path, reference.get_pixel_format(), reference.get_gamma(),
                                            Cogwheel::Math::Vector2ui(reference.get_width(), reference.get_height()));
        images.push_back(diff_image);

        printf("Compare '%s' with '%s'\n", reference_path.c_str(), target_path.c_str());

        switch (algorithm) {
        case Algorithm::RMS: {
            float rms = ImageOperations::Compare::rms(reference, target, diff_image);
            printf("  RMS: %f - lower is better.\n", rms);
            break;
        }
        case Algorithm::SSIM: {
            float ssim = ImageOperations::Compare::mssim(reference, target, 5, diff_image);
            ssim = Cogwheel::Math::max(0.0f, ssim); // Clamp in case ssim becomes negative due to imprecision.
            printf("  1.0f - ssim: %f - lower is better.\n", 1.0f - ssim);
            break;
        }
        }

        if (!diff_path.empty())
            store_image(diff_image, diff_path);

        return images;
    }

    Comparer(std::vector<char*> args, Cogwheel::Core::Engine& engine)
        : m_selected_image_index(-1) {

        m_images = apply(args);
        if (m_images.size() == 0)
            m_images.push_back(create_error_image());

        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &m_tex_ID);
        glBindTexture(GL_TEXTURE_2D, m_tex_ID);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        engine.add_mutating_callback(Comparer::update, this);
    }

    void update(Cogwheel::Core::Engine& engine) {
        using namespace Cogwheel::Input;
        using namespace Cogwheel::Math;

        const Keyboard* const keyboard = engine.get_keyboard();

        int image_index = m_selected_image_index;
        if (keyboard->was_released(Keyboard::Key::Left))
            --image_index;
        if (keyboard->was_released(Keyboard::Key::Right))
            ++image_index;
        image_index = Cogwheel::Math::clamp(image_index, 0, int(m_images.size() - 1));

        // Update the texture and window title in case the index has changed.
        if (m_selected_image_index != image_index) {

            // Update window title.
            std::string title = "Komodo - " + m_images[image_index].get_name();
            engine.get_window().set_name(title);

            glBindTexture(GL_TEXTURE_2D, m_tex_ID);
            const GLint BASE_IMAGE_LEVEL = 0;
            const GLint NO_BORDER = 0;
            Image image = m_images[image_index];
            int width = image.get_width(), height = image.get_height();
            RGB* gamma_corrected_pixels = new RGB[image.get_pixel_count()];
            #pragma omp parallel for schedule(dynamic, 16)
            for (int i = 0; i < (int)image.get_pixel_count(); ++i) {
                int x = i % width, y = i / width;
                gamma_corrected_pixels[i] = gammacorrect(image.get_pixel(Vector2ui(x, y)).rgb(), 1.0f / 2.2f);
            }
            glTexImage2D(GL_TEXTURE_2D, BASE_IMAGE_LEVEL, GL_RGB, width, height, NO_BORDER, GL_RGB, GL_FLOAT, gamma_corrected_pixels);

            m_selected_image_index = image_index;
        }

        render_image(engine.get_window(), m_tex_ID);
    }

    static void update(Cogwheel::Core::Engine& engine, void* comparer) {
        ((Comparer*)comparer)->update(engine);
    }

private:
    std::vector<Image> m_images;
    int m_selected_image_index;

    GLuint m_tex_ID = 0u;
};

#endif // _KOMODO_COMPARER_H_