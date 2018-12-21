// SimpleViewer material scene.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _SIMPLEVIEWER_MATERIAL_SCENE_H_
#define _SIMPLEVIEWER_MATERIAL_SCENE_H_

#include <Cogwheel/Scene/Camera.h>
#include <Cogwheel/Scene/SceneNode.h>

namespace ImGui { class ImGuiAdaptor; }

namespace Scenes {

void create_material_scene(Cogwheel::Scene::Cameras::UID camera_ID, Cogwheel::Scene::SceneNode root_node, ImGui::ImGuiAdaptor* imgui);

} // NS Scenes

#endif // _SIMPLEVIEWER_MATERIAL_SCENE_H_