#include <gtest/gtest.h>

#include <BSDFs/GGXTest.h>
#include <BSDFs/LambertTest.h>

#include <LightSources/SphereLightTest.h>

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
