#include <GLFWDriver.h>

#include <cstdio>
#include <iostream>

void initializer(Cogwheel::Core::Engine& engine) {
    std::cout << "Initialize baby!" << std::endl;
    engine.getWindow().setName("SimpleViewer");
}

void main(int argc, char** argv) {
    GLFWDriver::run(initializer);
}
