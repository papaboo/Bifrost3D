// Komodo Image Tool.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <Cogwheel/Core/Engine.h>

#include <GLFWDriver.h>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <GL/gl.h>
#undef RGB

#include <Comparer.h>

using namespace Cogwheel::Core;

// Global state
std::vector<char*> g_args;
Comparer* g_operation;

void update(Engine& engine, void* none) {
    assert(g_operation != nullptr);

    g_operation->update(engine);

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

        glBindTexture(GL_TEXTURE_2D, g_operation->get_texture_ID());
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
    return 0;
}

int window_initialized(Cogwheel::Core::Engine& engine, Cogwheel::Core::Window& window) {

    Images::allocate(3u);

    std::string operation_name = g_args[0];
    g_args.erase(g_args.begin());

    if (std::string(operation_name).compare("--compare") == 0)
        g_operation = new Comparer(g_args);
    else {
        g_operation = new Comparer(g_args); // TODO Null operation
        printf("Unrecognized argument: '%s'\n", operation_name.c_str());
    }

    engine.add_mutating_callback(update, nullptr);

    return 0;
}

void print_usage() {
    char* usage =
        "usage Komodo Image Tool:\n"
        "  -h | --help: Show command line usage for Komodo.\n"
        "  -l | --headless: Launch without opening a window.\n"
        "     | --compare: Perform image comparisons.\n";

    printf("%s", usage);
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
        GLFWDriver::run(initialize, window_initialized);
}
