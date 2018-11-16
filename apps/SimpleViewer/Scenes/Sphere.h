// SimpleViewer sphere scene.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _SIMPLEVIEWER_SPHERE_SCENE_H_
#define _SIMPLEVIEWER_SPHERE_SCENE_H_

#include <Cogwheel/Core/Engine.h>
#include <Cogwheel/Scene/Camera.h>
#include <Cogwheel/Scene/SceneRoot.h>

namespace ImGui { class ImGuiAdaptor; }

namespace Scenes {

void create_sphere_scene(Cogwheel::Core::Engine& engine, Cogwheel::Scene::Cameras::UID camera_ID, 
                         Cogwheel::Scene::SceneRoots::UID scene_ID, ImGui::ImGuiAdaptor* imgui);

} // NS Scenes

#endif // _SIMPLEVIEWER_SPHERE_SCENE_H_