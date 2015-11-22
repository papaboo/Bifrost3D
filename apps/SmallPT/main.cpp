// Smallpt viewer.
// ----------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

// Disable warning about fopen. We can live with it.
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <smallpt.h>

#include <Core/Array.h>

#include <glfw/glfw3.h>

#include <stdio.h>

using Cogwheel::Core::Array;
using namespace Cogwheel::Math;

inline float clamp(float x) { return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }
inline int toInt(float x) { return int(pow(clamp(x), 1.0f / 2.2f) * 255.0f + .5f); }

bool gRestartAccumulation = true;
Array<RGB> gBackbuffer = Array<RGB>(0); // TODO double3* backbuffer for precission and because the size is known.
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
            for (unsigned int i = 0; i < gBackbuffer.size(); ++i) {
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

void main(int argc, char** argv) {

    if (!glfwInit())
        exit(EXIT_FAILURE);

    GLFWwindow* window = glfwCreateWindow(256, 196, "smallpt", NULL, NULL);

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
    gBackbuffer = Array<RGB>(0);
    int accumulations = 0;
    GLuint texID; {
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &texID);
        glBindTexture(GL_TEXTURE_2D, texID);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }

    while (!glfwWindowShouldClose(window))
    {
        // Keep running
        glfwPollEvents();

        if (gRestartAccumulation) {
            gRestartAccumulation = false;
            accumulations = 0;
            glfwGetFramebufferSize(window, &gWindowWidth, &gWindowHeight);
            if (gWindowWidth * gWindowHeight != gBackbuffer.size()) {
                gBackbuffer.resize(gWindowWidth * gWindowHeight);
                for (RGB& p : gBackbuffer)
                    p = RGB::black();
            }
        }

        smallpt::accumulateRadiance(camera, gWindowWidth, gWindowHeight, gBackbuffer.data(), accumulations);

        { // Update the backbuffer.
            glViewport(0, 0, gWindowWidth, gWindowHeight);

            { // Setup matrices. I really don't need to do this every frame, since they never change.
                glMatrixMode(GL_PROJECTION);
                glLoadIdentity();
                glOrtho(-1, 1, -1.f, 1.f, 1.f, -1.f);

                glMatrixMode(GL_MODELVIEW);
                glLoadIdentity();
            }

            // TODO Logarithmic upload.

            glBindTexture(GL_TEXTURE_2D, texID);
            const GLint BASE_IMAGE_LEVEL = 0;
            const GLint noBorder = 0;
            glTexImage2D(GL_TEXTURE_2D, BASE_IMAGE_LEVEL, GL_RGB, gWindowWidth, gWindowHeight, noBorder, GL_RGB, GL_FLOAT, gBackbuffer.data());

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
