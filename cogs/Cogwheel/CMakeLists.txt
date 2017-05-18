SET(ASSETS_SRCS 
  Cogwheel/Assets/Image.h
  Cogwheel/Assets/Image.cpp
  Cogwheel/Assets/InfiniteAreaLight.h
  Cogwheel/Assets/Material.h
  Cogwheel/Assets/Material.cpp
  Cogwheel/Assets/Mesh.h
  Cogwheel/Assets/Mesh.cpp
  Cogwheel/Assets/MeshCreation.h
  Cogwheel/Assets/MeshCreation.cpp
  Cogwheel/Assets/MeshModel.h
  Cogwheel/Assets/MeshModel.cpp
  Cogwheel/Assets/Texture.h
  Cogwheel/Assets/Texture.cpp
)

SET(ASSETS_SHADING_SRCS 
  Cogwheel/Assets/Shading/BurleyRho.h
  Cogwheel/Assets/Shading/BurleyRho.png
  Cogwheel/Assets/Shading/DefaultShadingRho.h
  Cogwheel/Assets/Shading/DefaultShadingRho.png
  Cogwheel/Assets/Shading/GGXRho.h
  Cogwheel/Assets/Shading/GGXRho.png
  Cogwheel/Assets/Shading/GGXWithFresnelRho.h
  Cogwheel/Assets/Shading/GGXWithFresnelRho.png
  Cogwheel/Assets/Shading/OrenNayarRho.h
  Cogwheel/Assets/Shading/OrenNayarRho.png
)

SET(CORE_SRCS 
  Cogwheel/Core/Array.h
  Cogwheel/Core/Bitmask.h
  Cogwheel/Core/ChangeSet.h
  Cogwheel/Core/Engine.h
  Cogwheel/Core/Engine.cpp
  Cogwheel/Core/Iterable.h
  Cogwheel/Core/Renderer.h
  Cogwheel/Core/Renderer.cpp
  Cogwheel/Core/Time.h
  Cogwheel/Core/UniqueIDGenerator.h
  Cogwheel/Core/UniqueIDGenerator.impl
  Cogwheel/Core/Window.h
)

SET(INPUT_SRCS 
  Cogwheel/Input/Keyboard.h
  Cogwheel/Input/Mouse.h
)

SET(MATH_SRCS 
  Cogwheel/Math/AABB.h
  Cogwheel/Math/Color.h
  Cogwheel/Math/Conversions.h
  Cogwheel/Math/Constants.h
  Cogwheel/Math/Distribution1D.h
  Cogwheel/Math/Distribution2D.h
  Cogwheel/Math/Distributions.h
  Cogwheel/Math/Math.h
  Cogwheel/Math/Matrix.h
  Cogwheel/Math/Polynomial.h
  Cogwheel/Math/Quaternion.h
  Cogwheel/Math/Ray.h
  Cogwheel/Math/Rect.h
  Cogwheel/Math/RNG.h
  Cogwheel/Math/Statistics.h
  Cogwheel/Math/Transform.h
  Cogwheel/Math/Utils.h
  Cogwheel/Math/Vector.h
  Cogwheel/Math/VectorOperators.h
)

SET(SCENE_SRCS 
  Cogwheel/Scene/Camera.cpp
  Cogwheel/Scene/Camera.h
  Cogwheel/Scene/LightSource.cpp
  Cogwheel/Scene/LightSource.h
  Cogwheel/Scene/SceneNode.cpp
  Cogwheel/Scene/SceneNode.h
  Cogwheel/Scene/SceneRoot.cpp
  Cogwheel/Scene/SceneRoot.h
)

add_library(Cogwheel ${ASSETS_SRCS} ${ASSETS_SHADING_SRCS} ${CORE_SRCS} ${INPUT_SRCS} ${MATH_SRCS} ${SCENE_SRCS})

SOURCE_GROUP("Assets" FILES ${ASSETS_SRCS})
SOURCE_GROUP("Assets\\Shading" FILES ${ASSETS_SHADING_SRCS})
SOURCE_GROUP("Core" FILES ${CORE_SRCS})
SOURCE_GROUP("Input" FILES ${INPUT_SRCS})
SOURCE_GROUP("Math" FILES ${MATH_SRCS})
SOURCE_GROUP("Scene" FILES ${SCENE_SRCS})

target_include_directories(Cogwheel PUBLIC .)

set_target_properties(Cogwheel PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Cogs"
  COMPILE_FLAGS "-W3"
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast")