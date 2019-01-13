// Bifrost parallel utility functions.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2017, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. 
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_CORE_PARALLEL_H_
#define _BIFROST_CORE_PARALLEL_H_

#include <functional>

#include <omp.h>

namespace Bifrost {
namespace Core {
namespace Parallel {

template <typename LocalState>
void for_range(int begin, int end, std::function<LocalState()> local_init, 
               std::function<void(int, LocalState&)> body, 
               std::function<void(LocalState)> local_finally) {
#pragma omp parallel
    {
        auto local_state = local_init();

        int thread_id = omp_get_thread_num();
        int thread_count = omp_get_num_threads();
        for (int i = begin + thread_id; i < end; i += thread_count)
            body(i, local_state);

#pragma omp critical
        local_finally(local_state);
    }
}

} // NS Parallel
} // NS Core
} // NS Bifrost

#endif // _BIFROST_CORE_PARALLEL_H_
