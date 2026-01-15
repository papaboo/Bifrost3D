// Subsurface scattering testbed
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <Integrate.h>
#include <Plot.h>

enum class Command { Integrate, Plot };

struct Options {
    Command command = Command::Integrate;
    float slab_thickness = INFINITY;

    static Options parse(int argc, char** argv) {
        Options options;

        int argument = 1;
        while (argument < argc) {
            if (strcmp(argv[argument], "--integrate") == 0) {
                options.command = Command::Integrate;
            } else if (strcmp(argv[argument], "--plot") == 0) {
                options.command = Command::Plot;
            } else if (strcmp(argv[argument], "--thickness") == 0) {
                char* str_end;
                options.slab_thickness = strtof(argv[++argument], &str_end);
            }  else
                printf("Unknown argument: '%s'\n", argv[argument]);

            ++argument;
        }

        return options;
    }

    static void print_usage() {
        printf("Subsurface scattering testbed usage:\n"
            "  --integrate: Integrate SSS in a slab. This is the default command.\n"
            "  --plot: Plots from SSS in a slab.\n"
            "  --thickness <float>: Integrate SSS in a slab with the given thickness. Semiinfinite by default.\n");
    }
};

int main(int argc, char** argv) {
    std::string command = argc >= 2 ? std::string(argv[1]) : "";
    if (command.compare("-h") == 0 || command.compare("--help") == 0) {
        Options::print_usage();
        return 0;
    }

    Options options = Options::parse(argc, argv);

    printf("Subsurface scattering testbed with slab thickness %.3f\n", options.slab_thickness);

    Bifrost::Assets::Images::allocate(1u);

    if (options.command == Command::Integrate)
        Integrate::integrate(options.slab_thickness);
    else if (options.command == Command::Plot)
        Plot::plot(options.slab_thickness);

    return 0;
}