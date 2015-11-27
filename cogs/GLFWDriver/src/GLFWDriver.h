// Cogwheel GLFW main.
// ----------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _COGWHEEL_GLFW_DRIVER_H_
#define _COGWHEEL_GLFW_DRIVER_H_

#include <glfw/glfw3.h>

#include <string>

// TODO Move Window and Engine into Cogwheel when done with prototyping.
namespace Core {
class Window {
public:
    Window(const std::string& name, int width, int height)
        : mName(name), mWidth(width), mHeight(height) { }
    
    inline std::string getName() const { return mName; }
    inline void setName(const std::string& name) { mName = name; }

    inline int getWidth() const { return mWidth; }
    inline int getHeight() const { return mHeight; }

    inline int resize(int width, int height) {
        mWidth = width; mHeight = height;
    }

private:
    std::string mName;
    int mWidth, mHeight;
};

// TODO Make it a singleton.
class Engine {
public:
    Engine()
        : mWindow(Window("Cogwheel", 640, 480))
        , mQuit(false)
        , mIterations(0) { }

    inline Window& getWindow() { return mWindow; }

    inline bool quitRequested() const { return mQuit; }

    void doLoop(float dt) {
        // TODO Time struct with smooth delta time as well
        printf("dt: %f\n", dt);

        // TODO Invoke modules.

        ++mIterations;
    }

private:
    Window mWindow;
    bool mQuit;
    unsigned int mIterations;
};
}

namespace GLFWDriver {

    template <typename Initializer>
    void run(Initializer& initializer) {
        if (!glfwInit())
            exit(EXIT_FAILURE);

        // Create engine.
        Core::Engine engine = Core::Engine(); // TODO Make it a global inside the translation unit / cpp file? Then it could be used by all GLFW callbacks without hacing to pass it in.
        initializer(engine);

        GLFWwindow* window = glfwCreateWindow(engine.getWindow().getWidth(), engine.getWindow().getHeight(), engine.getWindow().getName().c_str(), NULL, NULL);

        if (!window) {
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);
        //glfwSetKeyCallback(window, keyCallback);
        //glfwSetWindowSizeCallback(window, windowSizeCallback);

        double previousTime = glfwGetTime();

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            // Poll time
            double currentTime = glfwGetTime();
            float deltaTime = float(currentTime - previousTime);
            previousTime = currentTime;
            
            engine.doLoop(deltaTime);

            glfwSwapBuffers(window);

            if (engine.quitRequested())
                glfwSetWindowShouldClose(window, GL_TRUE);
        }

        glfwDestroyWindow(window);
        glfwTerminate();
    }

} // NS GLFWDriver

#endif // _COGWHEEL_GLFW_DRIVER_H_