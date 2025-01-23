// SimpleViewer scene utilities.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _SIMPLEVIEWER_SCENE_UTILITIES_H_
#define _SIMPLEVIEWER_SCENE_UTILITIES_H_

#include <Bifrost/Assets/Material.h>
#include <Bifrost/Scene/SceneNode.h>

#include <filesystem>

namespace Scenes {

Bifrost::Scene::SceneNode create_checkered_floor(float floor_size, float checker_size);

void replace_material(Bifrost::Assets::Material material, Bifrost::Scene::SceneNode parent_node, const std::string& child_scene_node_name);

Bifrost::Scene::SceneNode load_shader_ball(const std::filesystem::path& data_directory, Bifrost::Assets::Material material);

} // NS Scenes

#endif // _SIMPLEVIEWER_SCENE_UTILITIES_H_