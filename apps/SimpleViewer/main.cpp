#include <GLFWDriver.h>

#include <cstdio>
#include <iostream>

void initializer(Cogwheel::Core::Engine& engine) {
    std::cout << "Initialize baby!" << std::endl;
    engine.get_window().set_name("SimpleViewer");
}

void main(int argc, char** argv) {
    GLFWDriver::run(initializer);
}
