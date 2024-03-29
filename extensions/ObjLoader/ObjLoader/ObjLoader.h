// Bifrost obj model loader.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_OBJ_LOADER_H_
#define _BIFROST_ASSETS_OBJ_LOADER_H_

#include <Bifrost/Utils/IdDeclarations.h>

#include <string>

namespace ObjLoader {

typedef Bifrost::Assets::ImageID (*ImageLoader)(const std::string& filename);

// -----------------------------------------------------------------------
// Loads an obj file.
// Future work:
// * Return an (optional) list of created mesh model IDs?
// * Reserve capacity for Mesh, MeshModels and SceneNodes before creating them.
// -----------------------------------------------------------------------
Bifrost::Scene::SceneNodeID load(const std::string& filename, ImageLoader image_loader);

bool file_supported(const std::string& filename);

} // NS ObjLoader

#endif // _BIFROST_ASSETS_OBJ_LOADER_H_
