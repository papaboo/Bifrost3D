// OptiXRenderer testing utils.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Utils.h>

#include <windows.h>
#undef RGB

std::filesystem::path get_data_directory() {
    char application_path[512];
    GetModuleFileName(nullptr, application_path, 512);
    auto application_directory = std::filesystem::path(application_path).parent_path();
    return application_directory.parent_path() / "Data";
}