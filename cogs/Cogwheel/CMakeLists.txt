include_directories(.)

SET(ASSETS_SRCS 
  Cogwheel/Assets/Material.h
  Cogwheel/Assets/Material.cpp
  Cogwheel/Assets/Mesh.h
  Cogwheel/Assets/Mesh.cpp
  Cogwheel/Assets/MeshCreation.h
  Cogwheel/Assets/MeshCreation.cpp
  Cogwheel/Assets/MeshModel.h
  Cogwheel/Assets/MeshModel.cpp
)

SET(CORE_SRCS 
  Cogwheel/Core/Array.h
  Cogwheel/Core/Engine.h
  Cogwheel/Core/Engine.cpp
  Cogwheel/Core/Iterable.h
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
  Cogwheel/Math/Math.h
  Cogwheel/Math/Matrix.h
  Cogwheel/Math/Quaternion.h
  Cogwheel/Math/Ray.h
  Cogwheel/Math/Rect.h
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
)

add_library(Gogwheel ${ASSETS_SRCS} ${CORE_SRCS} ${INPUT_SRCS} ${MATH_SRCS} ${SCENE_SRCS})

SOURCE_GROUP("Assets" FILES ${ASSETS_SRCS})
SOURCE_GROUP("Core" FILES ${CORE_SRCS})
SOURCE_GROUP("Input" FILES ${INPUT_SRCS})
SOURCE_GROUP("Math" FILES ${MATH_SRCS})
SOURCE_GROUP("Scene" FILES ${SCENE_SRCS})

set_target_properties(Gogwheel PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Cogs"
  COMPILE_FLAGS "-W3"
)