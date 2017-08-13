// Komodo image diffing.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _KOMODO_DIFF_H_
#define _KOMODO_DIFF_H_

#include <Utils.h>

#include <ImageOperations/Diff.h>

#include <vector>

using namespace Cogwheel::Assets;

class Diff {
public:

    enum class Algorithm { SSIM, RMS, Mean };

    static std::vector<Image> apply(std::vector<char*> args) {

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
            else if (arg.compare("--mean") == 0)
                algorithm = Algorithm::Mean;
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

        Image diff_image = Image();
        if (!diff_path.empty()) {
            diff_image = Images::create2D(diff_path, reference.get_pixel_format(), reference.get_gamma(),
                                          Cogwheel::Math::Vector2ui(reference.get_width(), reference.get_height()));
            images.push_back(diff_image);
        }

        printf("Diff '%s' with '%s'\n", reference_path.c_str(), target_path.c_str());

        switch (algorithm) {
        case Algorithm::Mean: {
            float mean = ImageOperations::Diff::mean(reference, target, diff_image);
            printf("  mean: %f - lower is better.\n", mean);
            break;
        }
        case Algorithm::RMS: {
            float rms = ImageOperations::Diff::rms(reference, target, diff_image);
            printf("  RMS: %f - lower is better.\n", rms);
            break;
        }
        case Algorithm::SSIM: {
            float ssim = ImageOperations::Diff::ssim(reference, target, diff_image);
            printf("  ssim: %f - higher is better.\n", ssim);
            break;
        }
        }

        return images;
    }

};

#endif // _KOMODO_DIFF_H_