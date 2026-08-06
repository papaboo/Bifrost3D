set(PROJECT_NAME "SubsurfaceScatteringTestBed")

set(SRCS Integrate.h main.cpp Plot.h Utils.h)

add_executable(${PROJECT_NAME} ${SRCS})

target_include_directories(${PROJECT_NAME} PRIVATE .)

target_link_libraries(${PROJECT_NAME} Bifrost StbImageWriter)

source_group("" FILES ${SRCS})

set_target_properties(${PROJECT_NAME} PROPERTIES
  FOLDER "Apps/Dev"
)

target_compile_options(${PROJECT_NAME} PRIVATE ${CXX_FAST_MATH})