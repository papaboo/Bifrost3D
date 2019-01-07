// Cogwheel mesh creation utilities.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_ASSETS_MESH_CREATION_H_
#define _COGWHEEL_ASSETS_MESH_CREATION_H_

#include <Cogwheel/Assets/Mesh.h>

namespace Cogwheel {
namespace Assets {

//----------------------------------------------------------------------------
// Mesh creation utilities.
// Future work
// * Tex coord functions and pass them to the relevant mesh creators.
// * Allow for non uniform scaling on creation, 
//   since the transforms only support uniform scaling.
//----------------------------------------------------------------------------
namespace MeshCreation {

Meshes::UID plane(unsigned int quads_pr_side, MeshFlags buffer_bitmask = MeshFlag::AllBuffers);

Meshes::UID cube(unsigned int quads_pr_side, Math::Vector3f scaling = Math::Vector3f::one(), MeshFlags buffer_bitmask = MeshFlag::AllBuffers);

Meshes::UID cylinder(unsigned int vertical_quads, unsigned int circumference_quads, MeshFlags buffer_bitmask = MeshFlag::AllBuffers);

Meshes::UID revolved_sphere(unsigned int longitude_quads, unsigned int latitude_quads, MeshFlags buffer_bitmask = MeshFlag::AllBuffers);

Meshes::UID torus(unsigned int revolution_quads, unsigned int circumference_quads, float minor_radius, MeshFlags buffer_bitmask = MeshFlag::AllBuffers);

} // NS MeshCreation
} // NS Assets
} // NS Cogwheel

#endif // _COGWHEEL_ASSETS_MESH_CREATION_H_