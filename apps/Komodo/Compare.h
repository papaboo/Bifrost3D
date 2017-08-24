// Komodo image comparison.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _KOMODO_COMPARE_H_
#define _KOMODO_COMPARE_H_

#include <Utils.h>

#include <ImageOperations/Compare.h>

#include <vector>

using namespace Cogwheel::Assets;

class Compare {
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
            printf("  error: Could not load reference '%s'\n", reference_path.c_str());
            return std::vector<Image>();
        } else
            images.push_back(reference);

        Image target = load_image(target_path);
        if (!target.exists()) {
            printf("  error: Could not load target '%s'\n", target_path.c_str());
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

};

#endif // _KOMODO_COMPARE_H_