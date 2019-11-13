SET(SRCS 
  src/gtest.cc
  src/gtest-death-test.cc
  src/gtest-filepath.cc
  src/gtest-internal-inl.h
  src/gtest-matchers.cc
  src/gtest-port.cc
  src/gtest-printers.cc
  src/gtest-test-part.cc
  src/gtest-typed-test.cc
)

set(HEADERS
  gtest/internal/gtest-internal.h
  gtest/internal/gtest-string.h
  gtest/gtest.h
  gtest/gtest_pred_impl.h
  gtest/gtest_prod.h
  gtest/gtest-death-test.h
  gtest/gtest-matchers.h
  gtest/gtest-message.h
  gtest/gtest-param-test.h
  gtest/gtest-printers.h
  gtest/gtest-spi.h
  gtest/gtest-test-part.h
  gtest/gtest-typed-test.h
)

set(INTERNAL_HEADERS
  gtest/internal/gtest-death-test-internal.h
  gtest/internal/gtest-death-test-internal.h
  gtest/internal/gtest-internal.h
  gtest/internal/gtest-param-util.h
  gtest/internal/gtest-port.h
  gtest/internal/gtest-port-arch.h
  gtest/internal/gtest-string.h
  gtest/internal/gtest-type-util.h
  gtest/internal/gtest-type-util.h.pump
)

source_group("src" FILES ${SRCS})
source_group("gtest" FILES ${HEADERS})
source_group("gtest\\internal" FILES ${INTERNAL_HEADERS})

add_library(gtest ${SRCS} ${HEADERS} ${INTERNAL_HEADERS})

target_include_directories(gtest PUBLIC .)

set_target_properties(gtest PROPERTIES
  LINKER_LANGUAGE CXX
  FOLDER "Libs"
)