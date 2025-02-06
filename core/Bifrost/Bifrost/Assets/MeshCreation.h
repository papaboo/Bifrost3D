// Bifrost mesh creation utilities.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_MESH_CREATION_H_
#define _BIFROST_ASSETS_MESH_CREATION_H_

#include <Bifrost/Assets/Mesh.h>

namespace Bifrost::Assets {

//----------------------------------------------------------------------------
// Mesh creation utilities.
//----------------------------------------------------------------------------
namespace MeshCreation {

MeshID plane(unsigned int quads_per_edge, MeshFlags buffer_bitmask = MeshFlag::AllBuffers);

MeshID box(unsigned int quads_per_edge, Math::Vector3f size = Math::Vector3f::one(), MeshFlags buffer_bitmask = MeshFlag::AllBuffers);

MeshID beveled_box(unsigned int quads_per_side, float normalized_bevel_size, Math::Vector3f size = Math::Vector3f::one(), MeshFlags buffer_bitmask = MeshFlag::AllBuffers);

MeshID cylinder(unsigned int vertical_quads, unsigned int circumference_quads, MeshFlags buffer_bitmask = MeshFlag::AllBuffers);

MeshID revolved_sphere(unsigned int longitude_quads, unsigned int latitude_quads, MeshFlags buffer_bitmask = MeshFlag::AllBuffers);

MeshID spherical_box(unsigned int quads_per_edge, MeshFlags buffer_bitmask = MeshFlag::AllBuffers);

MeshID torus(unsigned int revolution_quads, unsigned int circumference_quads, float minor_radius, MeshFlags buffer_bitmask = MeshFlag::AllBuffers);

} // NS MeshCreation
} // NS Bifrost::Assets

#endif // _BIFROST_ASSETS_MESH_CREATION_H_
