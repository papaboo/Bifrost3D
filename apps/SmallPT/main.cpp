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

#include <Bifrost/Core/Array.h>

#include <glfw/glfw3.h>

#include <stdio.h>

using Bifrost::Core::Array;
using namespace Bifrost::Math;

inline float clamp(float x) { return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }
inline int toInt(float x) { return int(pow(clamp(x), 1.0f / 2.2f) * 255.0f + .5f); }

bool gRestartAccumulation = true;
RGB* gBackbuffer = nullptr;
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
                RGB& c = gBackbuffer[i];
                fprintf(f, "%d %d %d ", toInt(c.r), toInt(c.g), toInt(c.b));
            }
        }
        break;
    }
}

void windowSizeCallback(GLFWwindow* window, int width, int height) {
    // Backbuffer and window sizes are updated by the re-initialization in the main loop.
    gRestartAccumulation = true;
}

inline bool is_power_of_two_or_zero(unsigned int v) {
    return (v & (v - 1)) == 0;
}

void main(int argc, char** argv) {

    if (!glfwInit())
        exit(EXIT_FAILURE);

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
    smallpt::Ray camera(Vector3d(50, 52, 295.6), normalize(Vector3d(0, -0.042612, -1))); // cam pos, dir
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
            }
        }

        smallpt::accumulateRadiance(camera, gWindowWidth, gWindowHeight, gBackbuffer, accumulations);

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
            // Divide by four to get the first 8 frames.
            int INTERACTIVE_FRAMES = 3;
            if (accumulations < INTERACTIVE_FRAMES || is_power_of_two_or_zero(accumulations - INTERACTIVE_FRAMES)) {
                glBindTexture(GL_TEXTURE_2D, tex_ID);
                const GLint BASE_IMAGE_LEVEL = 0;
                const GLint NO_BORDER = 0;
                glTexImage2D(GL_TEXTURE_2D, BASE_IMAGE_LEVEL, GL_RGB, gWindowWidth, gWindowHeight, NO_BORDER, GL_RGB, GL_FLOAT, gBackbuffer);
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
