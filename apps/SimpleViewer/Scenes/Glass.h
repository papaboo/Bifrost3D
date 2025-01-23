// SimpleViewer glass scene.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _SIMPLEVIEWER_GLASS_SCENE_H_
#define _SIMPLEVIEWER_GLASS_SCENE_H_

#include <Bifrost/Scene/Camera.h>
#include <Bifrost/Scene/SceneNode.h>

namespace Scenes {

void create_glass_scene(Bifrost::Scene::CameraID camera_ID, Bifrost::Scene::SceneNode root_node, const std::filesystem::path& data_directory);

} // NS Scenes

#endif // _SIMPLEVIEWER_GLASS_SCENE_H_