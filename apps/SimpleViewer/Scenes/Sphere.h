// SimpleViewer sphere scene.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _SIMPLEVIEWER_SPHERE_SCENE_H_
#define _SIMPLEVIEWER_SPHERE_SCENE_H_

#include <Bifrost/Core/Engine.h>
#include <Bifrost/Scene/Camera.h>
#include <Bifrost/Scene/SceneRoot.h>

namespace ImGui { class ImGuiAdaptor; }

namespace Scenes {

void create_sphere_scene(Bifrost::Core::Engine& engine, Bifrost::Scene::Cameras::UID camera_ID, 
                         Bifrost::Scene::SceneRoots::UID scene_ID, ImGui::ImGuiAdaptor* imgui);

} // NS Scenes

#endif // _SIMPLEVIEWER_SPHERE_SCENE_H_