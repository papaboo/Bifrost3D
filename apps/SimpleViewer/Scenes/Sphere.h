// SimpleViewer sphere scene.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _SIMPLEVIEWER_SPHERE_SCENE_H_
#define _SIMPLEVIEWER_SPHERE_SCENE_H_

#include <Bifrost/Scene/Camera.h>
#include <Bifrost/Scene/SceneNode.h>

namespace Scenes {

void create_sphere_scene(Bifrost::Scene::CameraID camera_ID, Bifrost::Scene::SceneNode root_node);

} // NS Scenes

#endif // _SIMPLEVIEWER_SPHERE_SCENE_H_