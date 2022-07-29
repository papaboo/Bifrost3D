// Bifrost ID forward declarations.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_UTILS_ID_DECLARATIONS_H_
#define _BIFROST_UTILS_ID_DECLARATIONS_H_

#include <Bifrost/Core/UniqueIDGenerator.h>

//-------------------------------------------------------------------------------------------------
// Resource ID forward declaration.
//-------------------------------------------------------------------------------------------------
namespace Bifrost::Assets {
class Images;
typedef Core::TypedUIDGenerator<Images>::UID ImageID;
class Materials;
typedef Core::TypedUIDGenerator<Materials>::UID MaterialID;
class Meshes;
typedef Core::TypedUIDGenerator<Meshes>::UID MeshID;
class MeshModels;
typedef Core::TypedUIDGenerator<MeshModels>::UID MeshModelID;
class Textures;
typedef Core::TypedUIDGenerator<Textures>::UID TextureID;
}

namespace Bifrost::Core {
class Renderers;
typedef TypedUIDGenerator<Renderers>::UID RendererID;
}

namespace Bifrost::Scene {
class Cameras;
typedef Core::TypedUIDGenerator<Cameras>::UID CameraID;
class LightSources;
typedef Core::TypedUIDGenerator<LightSources>::UID LightSourceID;
class SceneNodes;
typedef Core::TypedUIDGenerator<SceneNodes>::UID SceneNodeID;
class SceneRoots;
typedef Core::TypedUIDGenerator<SceneRoots>::UID SceneRootID;
}

#endif // _BIFROST_UTILS_ID_DECLARATIONS_H_