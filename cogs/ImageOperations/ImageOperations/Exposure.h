// Image exposure oprations.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015-2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. 
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _IMAGE_OPERATIONS_EXPOSURE_H_
#define _IMAGE_OPERATIONS_EXPOSURE_H_

#include <Cogwheel/Assets/Image.h>

namespace ImageOperations {
namespace Exposure {

float average_log_luminance(Cogwheel::Assets::Images::UID image_ID);

} // NS Exposure
} // NS ImageOperations

#endif // _IMAGE_OPERATIONS_EXPOSURE_H_