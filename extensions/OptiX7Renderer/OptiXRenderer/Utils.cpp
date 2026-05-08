// OptiX renderer utilities for working with OptiX host code.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Utils.h>

#include <optix.h>

#include <string>
#include <stdexcept>

namespace OptiXRenderer {

void optix_check_error(OptixResult error, const char* const file, int line) {
    if (error != OPTIX_SUCCESS) {
        std::string message = "[file:" + std::string(file) + " line:" + std::to_string(line) + "] OptiX error: " + optixGetErrorName(error);
        printf("%s.\n", message.c_str());
        throw std::runtime_error(message);
    }
}

} // NS OptiXRenderer