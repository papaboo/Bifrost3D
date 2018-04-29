// DirectX 11 renderer prefix sum using compute.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2017, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_PREFIX_SUM_H_
#define _DX11RENDERER_RENDERER_PREFIX_SUM_H_

#include <DX11Renderer/Utils.h>

namespace DX11Renderer {

// ------------------------------------------------------------------------------------------------
// A prefix sum operator using compute shaders
// Future work:
// * Extend to handle ints, uints and floats. 
// * Support more than 256 * 256 elements.
// * Support SRVs.
// * Perhaps even handle arbitrary types by string concatenation / macros and runtime compilatation.
// ------------------------------------------------------------------------------------------------
class PrefixSum {
    const unsigned int GROUP_SIZE = 256;

    static inline unsigned int ceil_divide(unsigned int a, unsigned int b) {
        return (a / b) + ((a % b) > 0);
    }

public:

    PrefixSum(ID3D11Device1& device, const std::wstring& shader_folder_path) {
        OBlob reduce_shader_blob = compile_shader(shader_folder_path + L"Compute\\PrefixSum.hlsl", "cs_5_0", "reduce");
        THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(reduce_shader_blob), nullptr, &m_reduce_shader));

        OBlob downsweep_shader_blob = compile_shader(shader_folder_path + L"Compute\\PrefixSum.hlsl", "cs_5_0", "downsweep");
        THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(downsweep_shader_blob), nullptr, &m_downsweep_shader));

        int4 outer_constants = { 1, 0, 0, 0 };
        THROW_ON_FAILURE(create_constant_buffer(device, outer_constants, &m_outer_constants));

        int4 outer_single_iteration_constants = { 1, 1, 0, 0 };
        THROW_ON_FAILURE(create_constant_buffer(device, outer_single_iteration_constants, &m_outer_single_iteration_constants));

        int4 inner_constants = { int(GROUP_SIZE), 1, 1, 0 };
        THROW_ON_FAILURE(create_constant_buffer(device, inner_constants, &m_inner_constants));
    }

    void apply(ID3D11Device1& device, unsigned int* begin, unsigned int* end) {
        int element_count = int(end - begin);

        OUnorderedAccessView buffer_UAV;
        OBuffer gpu_buffer = create_default_buffer(device, DXGI_FORMAT_R32_UINT, begin, element_count, nullptr, &buffer_UAV);

        ID3D11DeviceContext1* context;
        device.GetImmediateContext1(&context);

        apply(*context, buffer_UAV, element_count);

        Readback::buffer(&device, context, gpu_buffer, begin, end);
    }

    void apply(ID3D11DeviceContext1& context, ID3D11UnorderedAccessView* buffer_UAV, unsigned int element_count = 0xFFFFFFFF) { 
        context.CSSetUnorderedAccessViews(0, 1, &buffer_UAV, 0u);

        bool run_two_iterations = element_count > GROUP_SIZE;
        OBuffer& outer_constants = run_two_iterations ? m_outer_constants : m_outer_single_iteration_constants;

        // Reduce
        context.CSSetShader(m_reduce_shader, nullptr, 0);
        context.CSSetConstantBuffers(0, 1, &outer_constants);
        context.Dispatch(ceil_divide(element_count, GROUP_SIZE), 1u, 1u);

        // Reduce and downsweep part two if needed.
        if (run_two_iterations) {
            // Inner reduction
            context.CSSetConstantBuffers(0, 1, &m_inner_constants);
            context.Dispatch(1u, 1u, 1u);

            // Inner downsweep
            context.CSSetShader(m_downsweep_shader, nullptr, 0);
            context.Dispatch(1u, 1u, 1u);

            // Outer downsweep.
            context.CSSetConstantBuffers(0, 1, &m_outer_constants);
            context.Dispatch(ceil_divide(element_count, GROUP_SIZE), 1u, 1u);

        } else {
            // Downsweep
            context.CSSetShader(m_downsweep_shader, nullptr, 0);
            context.Dispatch(ceil_divide(element_count, GROUP_SIZE), 1u, 1u);
        }

        // Cleanup to avoid resources bound to multiple registers.
        ID3D11UnorderedAccessView* null_UAV = nullptr;
        context.CSSetUnorderedAccessViews(0, 1, &null_UAV, nullptr);
    }

private:
    OComputeShader m_reduce_shader;
    OComputeShader m_downsweep_shader;

    OBuffer m_outer_constants;
    OBuffer m_outer_single_iteration_constants;
    OBuffer m_inner_constants;
};

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_PREFIX_SUM_H_