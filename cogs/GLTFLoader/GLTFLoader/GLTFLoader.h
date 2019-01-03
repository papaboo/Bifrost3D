// Cogwheel gltf model loader.
// ------------------------------------------------------------------------------------------------
// Copyright (C), Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _COGWHEEL_ASSETS_GLTF_LOADER_H_
#define _COGWHEEL_ASSETS_GLTF_LOADER_H_

#include <Cogwheel/Scene/SceneNode.h>
#include <string>

namespace GLTFLoader {

// ------------------------------------------------------------------------------------------------
// Loads a gltf file.
// Future work:
// * Support doubleSided on meshes. Requires mesh support first.
// * Reserve capacity for Mesh, MeshModels and SceneNodes before creating them.
// * Import cameras. Perhaps as viewpoints, since there is no way to enable/disable cameras.
// * Support triangle fan and triangle strip as well.
// ------------------------------------------------------------------------------------------------
Cogwheel::Scene::SceneNodes::UID load(const std::string& filename);

} // NS GLTFLoader

#endif // _COGWHEEL_ASSETS_GLTF_LOADER_H_