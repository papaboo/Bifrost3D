// Smallpt viewer.
// ----------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

// Disable warning about fopen. We can live with it.
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <smallpt.h>
#include <smallvpt.h>

#include <Bifrost/Math/Color.h>

#include <glfw/glfw3.h>

#include <stdio.h>

using namespace Bifrost::Math;

bool gRestartAccumulation = true;
RGB* gBackbuffer = nullptr;
RGB* sRGB_colors = nullptr;
int gWindowWidth = 0;
int gWindowHeight = 0;

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    switch (key) {
    case GLFW_KEY_ESCAPE:
        if (action == GLFW_RELEASE)
            glfwSetWindowShouldClose(window, GL_TRUE);
        break;
    case GLFW_KEY_ENTER:
        if (action == GLFW_RELEASE)
            gRestartAccumulation = true;
        break;
    case GLFW_KEY_SPACE:
        // Pause accumulation (and allow the user to single step through accumulations using arrows).
        ;
    case GLFW_KEY_P:
        if (action == GLFW_RELEASE) {
            // Write image to PPM file.
            FILE *f = fopen("image.ppm", "w");
            fprintf(f, "P3\n%d %d\n%d\n", gWindowWidth, gWindowHeight, 255);
            for (int i = 0; i < gWindowWidth * gWindowHeight; ++i) {
                RGB linear_color = gBackbuffer[i];
                RGB sRGB_color = Bifrost::Math::linear_to_sRGB(linear_color);
                RGB24 sRGB_24 = { sRGB_color.r, sRGB_color.g, sRGB_color.b };
                fprintf(f, "%d %d %d ", sRGB_24.r.raw, sRGB_24.g.raw, sRGB_24.b.raw);
            }
        }
        break;
    }
}

void windowSizeCallback(GLFWwindow* window, int width, int height) {
    // Avoid restarting accumulation if the window is minimized.
    bool is_minimized = width <= 0 || height <= 0;
    bool no_resize = gWindowWidth == width && gWindowHeight == height;
    if (is_minimized || no_resize)
        return;

    // Backbuffer and window sizes are updated by the re-initialization in the main loop.
    gRestartAccumulation = true;
}

inline bool is_power_of_two_or_zero(unsigned int v) {
    return (v & (v - 1)) == 0;
}

void main(int argc, char** argv) {

    printf("SmallPT: Use '--volumetric' argument to enable volumetric effects.\n");

    if (!glfwInit())
        exit(EXIT_FAILURE);

    bool volumetric_integrator = argc >= 2 && strcmp(argv[1], "--volumetric") == 0;

    GLFWwindow* window = glfwCreateWindow(256, 196, "smallpt", nullptr, nullptr);

    if (!window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetWindowSizeCallback(window, windowSizeCallback);

    // Initialization
    gRestartAccumulation = true;
    int accumulations = 0;
    GLuint tex_ID; {
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &tex_ID);
        glBindTexture(GL_TEXTURE_2D, tex_ID);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    }

    while (!glfwWindowShouldClose(window))
    {
        // Keep running
        glfwPollEvents();

        if (gRestartAccumulation) {
            gRestartAccumulation = false;
            accumulations = 0;
            int pixelCount = gWindowWidth * gWindowHeight;
            glfwGetFramebufferSize(window, &gWindowWidth, &gWindowHeight);
            if (gWindowWidth * gWindowHeight > int(pixelCount)) {
                delete[] gBackbuffer;
                gBackbuffer = new RGB[gWindowWidth * gWindowHeight];
                for (RGB* p = gBackbuffer; p < (gBackbuffer + gWindowWidth * gWindowHeight); ++p)
                    *p = RGB::black();

                delete[] sRGB_colors;
                sRGB_colors = new RGB[gWindowWidth * gWindowHeight];
            }
        }

        if (volumetric_integrator)
            smallvpt::accumulate_radiance(gWindowWidth, gWindowHeight, gBackbuffer, accumulations);
        else
            smallpt::accumulate_radiance(gWindowWidth, gWindowHeight, gBackbuffer, accumulations);

        { // Update the backbuffer.
            glViewport(0, 0, gWindowWidth, gWindowHeight);

            { // Setup matrices. I really don't need to do this every frame, since they never change.
                glMatrixMode(GL_PROJECTION);
                glLoadIdentity();
                glOrtho(-1, 1, -1.f, 1.f, 1.f, -1.f);

                glMatrixMode(GL_MODELVIEW);
                glLoadIdentity();
            }

            // Logarithmic upload. Uploaded every time the number of accumulations is a power of two.
            int INTERACTIVE_FRAMES = 3;
            if (accumulations < INTERACTIVE_FRAMES || is_power_of_two_or_zero(accumulations - INTERACTIVE_FRAMES) || (accumulations - INTERACTIVE_FRAMES) % 32 == 0) {
                // Backbuffer data is interpreted as sRGB, so we need to convert from linear colors to sRGB.
                int pixel_count = gWindowWidth * gWindowHeight;
                for (int i = 0; i < pixel_count; ++i)
                    sRGB_colors[i] = Bifrost::Math::linear_to_sRGB(gBackbuffer[i]);

                glBindTexture(GL_TEXTURE_2D, tex_ID);
                const GLint BASE_IMAGE_LEVEL = 0;
                const GLint NO_BORDER = 0;
                glTexImage2D(GL_TEXTURE_2D, BASE_IMAGE_LEVEL, GL_RGB, gWindowWidth, gWindowHeight, NO_BORDER, GL_RGB, GL_FLOAT, sRGB_colors);
            }

            glClear(GL_COLOR_BUFFER_BIT);
            glBegin(GL_QUADS); {

                glTexCoord2f(0.0f, 1.0f);
                glVertex3f(-1.0f, -1.0f, 0.f);

                glTexCoord2f(1.0f, 1.0f);
                glVertex3f(1.0f, -1.0f, 0.f);

                glTexCoord2f(1.0f, 0.0f);
                glVertex3f(1.0f, 1.0f, 0.f);

                glTexCoord2f(0.0f, 0.0f);
                glVertex3f(-1.0f, 1.0f, 0.f);

            } glEnd();
        }

        glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}
