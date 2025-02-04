// SimpleViewer material scene.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _SIMPLEVIEWER_MATERIAL_SCENE_H_
#define _SIMPLEVIEWER_MATERIAL_SCENE_H_

#include <Bifrost/Scene/Camera.h>
#include <Bifrost/Scene/SceneNode.h>

#include <filesystem>

namespace ImGui { class ImGuiAdaptor; }

namespace Scenes {

void create_material_scene(Bifrost::Scene::CameraID camera_ID, Bifrost::Scene::SceneNode root_node, ImGui::ImGuiAdaptor* imgui, const std::filesystem::path& resource_directory);

} // NS Scenes

#endif // _SIMPLEVIEWER_MATERIAL_SCENE_H_