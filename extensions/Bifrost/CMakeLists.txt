SET(ASSETS_SRCS 
  Bifrost/Assets/Image.h
  Bifrost/Assets/Image.cpp
  Bifrost/Assets/InfiniteAreaLight.h
  Bifrost/Assets/InfiniteAreaLight.inl
  Bifrost/Assets/Material.h
  Bifrost/Assets/Material.cpp
  Bifrost/Assets/Mesh.h
  Bifrost/Assets/Mesh.cpp
  Bifrost/Assets/MeshCreation.h
  Bifrost/Assets/MeshCreation.cpp
  Bifrost/Assets/MeshModel.h
  Bifrost/Assets/MeshModel.cpp
  Bifrost/Assets/Texture.h
  Bifrost/Assets/Texture.cpp
)

SET(ASSETS_SHADING_SRCS 
  Bifrost/Assets/Shading/BurleyRho.cpp
  Bifrost/Assets/Shading/BurleyRho.png
  Bifrost/Assets/Shading/DefaultShadingRho.cpp
  Bifrost/Assets/Shading/DefaultShadingRho.png
  Bifrost/Assets/Shading/Fittings.h
  Bifrost/Assets/Shading/GGXRho.cpp
  Bifrost/Assets/Shading/GGXRho.png
  Bifrost/Assets/Shading/GGXSPTDFit.cpp
  Bifrost/Assets/Shading/GGXSPTDFit.png
  Bifrost/Assets/Shading/GGXWithFresnelRho.cpp
  Bifrost/Assets/Shading/GGXWithFresnelRho.png
  Bifrost/Assets/Shading/OrenNayarRho.cpp
  Bifrost/Assets/Shading/OrenNayarRho.png
)

SET(CORE_SRCS 
  Bifrost/Core/Array.h
  Bifrost/Core/Bitmask.h
  Bifrost/Core/ChangeSet.h
  Bifrost/Core/Engine.h
  Bifrost/Core/Engine.cpp
  Bifrost/Core/Iterable.h
  Bifrost/Core/Parallel.h
  Bifrost/Core/Renderer.h
  Bifrost/Core/Renderer.cpp
  Bifrost/Core/Time.h
  Bifrost/Core/UniqueIDGenerator.h
  Bifrost/Core/UniqueIDGenerator.impl
  Bifrost/Core/Window.h
)

SET(INPUT_SRCS 
  Bifrost/Input/Keyboard.h
  Bifrost/Input/Mouse.h
)

SET(MATH_SRCS 
  Bifrost/Math/AABB.h
  Bifrost/Math/CameraEffects.h
  Bifrost/Math/Color.h
  Bifrost/Math/Constants.h
  Bifrost/Math/Conversions.h
  Bifrost/Math/Distribution1D.h
  Bifrost/Math/Distribution2D.h
  Bifrost/Math/Distributions.h
  Bifrost/Math/half.h
  Bifrost/Math/Intersect.h
  Bifrost/Math/Math.h
  Bifrost/Math/Matrix.h
  Bifrost/Math/MortonEncode.h
  Bifrost/Math/NelderMead.h
  Bifrost/Math/OctahedralNormal.h
  Bifrost/Math/Plane.h
  Bifrost/Math/Polynomial.h
  Bifrost/Math/Quaternion.h
  Bifrost/Math/Ray.h
  Bifrost/Math/Rect.h
  Bifrost/Math/RNG.h
  Bifrost/Math/Statistics.h
  Bifrost/Math/Transform.h
  Bifrost/Math/Utils.h
  Bifrost/Math/Vector.h
  Bifrost/Math/VectorOperators.h
)

SET(SCENE_SRCS 
  Bifrost/Scene/Camera.cpp
  Bifrost/Scene/Camera.h
  Bifrost/Scene/LightSource.cpp
  Bifrost/Scene/LightSource.h
  Bifrost/Scene/SceneNode.cpp
  Bifrost/Scene/SceneNode.h
  Bifrost/Scene/SceneRoot.cpp
  Bifrost/Scene/SceneRoot.h
)

add_library(Bifrost ${ASSETS_SRCS} ${ASSETS_SHADING_SRCS} ${CORE_SRCS} ${INPUT_SRCS} ${MATH_SRCS} ${SCENE_SRCS})

SOURCE_GROUP("Assets" FILES ${ASSETS_SRCS})
SOURCE_GROUP("Assets\\Shading" FILES ${ASSETS_SHADING_SRCS})
SOURCE_GROUP("Core" FILES ${CORE_SRCS})
SOURCE_GROUP("Input" FILES ${INPUT_SRCS})
SOURCE_GROUP("Math" FILES ${MATH_SRCS})
SOURCE_GROUP("Scene" FILES ${SCENE_SRCS})

target_include_directories(Bifrost PUBLIC .)

set_target_properties(Bifrost PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Extensions"
  COMPILE_FLAGS "-W3"
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast")