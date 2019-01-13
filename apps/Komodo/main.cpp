// Komodo Image Tool.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <Bifrost/Core/Engine.h>

#include <GLFWDriver.h>

#include <Blurer.h>
#include <ColorGrader.h>
#include <Comparer.h>

using namespace Bifrost::Core;

// Global state
std::vector<char*> g_args;
void* g_operation;

void print_usage() {
    char* usage =
        "usage Komodo Image Tool:\n"
        "  -h | --help: Show command line usage for Komodo.\n"
        "  -l | --headless: Launch without opening a window.\n"
        "     | --blur: Blur image.\n"
        "     | --color-grade: Color grade image.\n"
        "     | --compare: Perform image comparisons.\n";

    printf("%s", usage);
}

int initialize(Engine& engine) {
    engine.get_window().set_name("Komodo");
    engine.get_window().resize(1280, 960);
    return 0;
}

int window_initialized(Engine& engine, Window& window) {

    Images::allocate(3u);

    std::string operation_name = g_args[0];
    g_args.erase(g_args.begin());

    if (std::string(operation_name).compare("--blur") == 0)
        g_operation = new Blurer(g_args, engine);
    else if (std::string(operation_name).compare("--color-grade") == 0)
        g_operation = new ColorGrader(g_args, engine);
    else if (std::string(operation_name).compare("--compare") == 0)
        g_operation = new Comparer(g_args, engine);
    else {
        printf("Unrecognized argument: '%s'\n", operation_name.c_str());
        print_usage();
        return 1;
    }

    return 0;
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
        Engine* engine = new Engine("");
        initialize(*engine);
        engine->get_window().resize(0, 0);
        window_initialized(*engine, engine->get_window());
    } else
        GLFWDriver::run(initialize, window_initialized);
}
