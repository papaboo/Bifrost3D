// Cogwheel obj model loader.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_ASSETS_OBJ_LOADER_H_
#define _COGWHEEL_ASSETS_OBJ_LOADER_H_

#include <Cogwheel/Assets/Image.h>
#include <Cogwheel/Scene/SceneNode.h>
#include <string>

namespace ObjLoader {

typedef Cogwheel::Assets::Images::UID (*ImageLoader)(const std::string& filename);

// -----------------------------------------------------------------------
// Loads an obj file.
// Future work:
// * Support for materials.
// * Pass in texture 2D loader function as argument.
// * Return an (optional) list of created mesh model IDs?
// * Reserve capacity for Mesh, MeshModels and SceneNodes before creating them.
// -----------------------------------------------------------------------
Cogwheel::Scene::SceneNodes::UID load(const std::string& filename, ImageLoader image_loader);

} // NS ObjLoader

#endif // _COGWHEEL_ASSETS_OBJ_LOADER_H_