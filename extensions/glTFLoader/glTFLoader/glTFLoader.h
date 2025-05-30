// Bifrost glTF model loader.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_GLTF_LOADER_H_
#define _BIFROST_ASSETS_GLTF_LOADER_H_

#include <Bifrost/Utils/IdDeclarations.h>

#include <string>

namespace Bifrost::Scene { class SceneNode; }

namespace glTFLoader {

// ------------------------------------------------------------------------------------------------
// Loads a glTF file.
// See the glTF spec at https://github.com/KhronosGroup/glTF/tree/master/specification
// Future work:
// * Reserve capacity for Mesh, MeshModels and SceneNodes before creating them.
// * Import cameras. Perhaps as viewpoints, since there is no way to enable/disable cameras.
// * Support triangle fan and triangle strip as well.
// ------------------------------------------------------------------------------------------------
Bifrost::Scene::SceneNode load(const std::string& filename);

bool file_supported(const std::string& filename);

} // NS glTFLoader

#endif // _BIFROST_ASSETS_GLTF_LOADER_H_
