// OptiX renderer ptx loader.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <OptiXRenderer/PtxLoader.h>

#include <nvrtc.h>

#include <fstream>
#include <vector>

using namespace std;

namespace OptiXRenderer::PtxLoader {

inline void throw_nvrtc_error(nvrtcResult error, const string& file, int line) {
    if (error != NVRTC_SUCCESS) {
        string message = "[file:" + file + " line:" + to_string(line) + "] NVRTC errror: " + string(nvrtcGetErrorString(error));
        printf("%s.\n", message.c_str());
        throw exception(message.c_str(), error);
    }
}
#define THROW_NVRTC_ERROR(error) throw_nvrtc_error(error, __FILE__,__LINE__)

static string read_source_from_file(const string& absolute_file_path) {
    string file_content;

    // Try to open file
    ifstream file(absolute_file_path.c_str(), ios::binary);
    if (file.good()) {
        // Found usable source file
        vector<char> buffer = vector<char>(istreambuf_iterator<char>(file), {});
        file_content.assign(buffer.begin(), buffer.end());
    }

    return file_content;
}

static string get_ptx_from_cu_source(const string& sample_name, const string& cu_source, const string& name) {
    // Create program
    nvrtcProgram program = 0;
    THROW_NVRTC_ERROR(nvrtcCreateProgram(&program, cu_source.c_str(), name.c_str(), 0, nullptr, nullptr));

    // Renderer, optix and cuda include directories
    vector<string> include_dirs;
    include_dirs.push_back(string("-I") + OPTIX_RENDERER_INCLUDE_DIR);
    include_dirs.push_back(string("-I") + OPTIX_INCLUDE_DIR);
    include_dirs.push_back(string("-I") + CUDA_INCLUDE_DIR);

    vector<const char*> compile_args;
    for (const std::string& dir : include_dirs)
        compile_args.push_back(dir.c_str());

    // Compile options inspired by OptiX 7.3 SDK
    compile_args.push_back("-std=c++17");
    compile_args.push_back("-arch");
    compile_args.push_back("compute_60"); // TODO Get actual GPU compute arch at runtime.
    compile_args.push_back("-use_fast_math");
    compile_args.push_back("-lineinfo");
    compile_args.push_back("-default-device");
    compile_args.push_back("-rdc");
    compile_args.push_back("true");
    compile_args.push_back("-D__x86_64");

    const nvrtcResult compilationStatus = nvrtcCompileProgram(program, (int)compile_args.size(), compile_args.data());

    size_t log_size = 0;
    THROW_NVRTC_ERROR(nvrtcGetProgramLogSize(program, &log_size));
    string nvrtc_log; nvrtc_log.resize(log_size);
    if (log_size > 1) {
        THROW_NVRTC_ERROR(nvrtcGetProgramLog(program, &nvrtc_log[0]));
        printf("%s\n", nvrtc_log.c_str());
    }
    if (compilationStatus != NVRTC_SUCCESS)
        throw exception(("NVRTC Compilation failed.\n" + nvrtc_log).c_str(), compilationStatus);

    // Retrieve PTX code
    size_t ptx_size = 0;
    THROW_NVRTC_ERROR(nvrtcGetPTXSize(program, &ptx_size));
    string ptx; ptx.resize(ptx_size);
    THROW_NVRTC_ERROR(nvrtcGetPTX(program, ptx.data()));

    // Cleanup
    THROW_NVRTC_ERROR(nvrtcDestroyProgram(&program));

    return ptx;
}

string load_ptx_using_nvrtc(const string& relative_shader_path) {
    string absolute_shader_path = string(OPTIX_RENDERER_INCLUDE_DIR) + "/OptiXRenderer/" + relative_shader_path;

    string cu_source = read_source_from_file(absolute_shader_path);

    return get_ptx_from_cu_source("", cu_source, "");
}

string load_precompiled_ptx(const string& relative_shader_path) {
    return "Loading precompiled ptx currently not supported.";
}

} // NS OptiXRenderer::PtxLoader