// DirectX 11 compositor.
//-------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
//-------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_COMPOSITOR_H_
#define _DX11RENDERER_COMPOSITOR_H_

#include <Cogwheel/Core/Renderer.h>
#include <Cogwheel/Core/Window.h>
#include <Cogwheel/Scene/Camera.h>

//-------------------------------------------------------------------------------------------------
// Forward declarations.
//-------------------------------------------------------------------------------------------------
namespace Cogwheel {
namespace Core {
class Engine;
class Window;
}
}
struct HWND__;
typedef HWND__* HWND;
struct ID3D11Device1;

#define D3D11_CREATE_DEVICE_NONE 0

namespace DX11Renderer {

//-------------------------------------------------------------------------------------------------
// Renderer interface.
// Future work:
// * Pass masked areas (rects) from overlapping (opaque) cameras in render, to help with masking and culling.
//-------------------------------------------------------------------------------------------------
class IRenderer {
public:
    virtual ~IRenderer() {}
    virtual Cogwheel::Core::Renderers::UID get_ID() const = 0;
    virtual void handle_updates() = 0;
    virtual void render(Cogwheel::Scene::Cameras::UID camera_ID, int width, int height) = 0;
};

typedef IRenderer*(*RendererCreator)(ID3D11Device1&, int width_hint, int height_hint, const std::wstring& data_folder_path);

//-------------------------------------------------------------------------------------------------
// Utility function to create a 'performant' DX11 device.
//-------------------------------------------------------------------------------------------------
template <typename T> class OwnedResourcePtr;
using ODevice1 = DX11Renderer::OwnedResourcePtr<ID3D11Device1>;
ODevice1 create_performant_device1(unsigned int create_device_flags = D3D11_CREATE_DEVICE_NONE);
ODevice1 create_performant_debug_device1();

//-------------------------------------------------------------------------------------------------
// DirectX 11 compositor.
// Composits the rendered images from various cameras attached to the window.
//-------------------------------------------------------------------------------------------------
class Compositor final {
public:

    struct Initialization {
        Compositor* compositor;
        IRenderer* renderer;
    };
    static Initialization initialize(HWND& hwnd, const Cogwheel::Core::Window& window, const std::wstring& data_folder_path, RendererCreator renderer_creator);
    ~Compositor();

    IRenderer* attach_renderer(RendererCreator renderer_creator);

    void render();

    bool uses_v_sync() const;
    void set_v_sync(bool use_v_sync);

private:

    Compositor(HWND& hwnd, const Cogwheel::Core::Window& window, const std::wstring& data_folder_path);

    // Delete copy constructors to avoid having multiple versions of the same Compositor.
    Compositor(Compositor& other) = delete;
    Compositor& operator=(const Compositor& rhs) = delete;

    // Pimpl the state to avoid exposing DirectX dependencies.
    class Implementation;
    Implementation* m_impl;
};

static inline void render_callback(const Cogwheel::Core::Engine& engine, void* compositor) {
    static_cast<Compositor*>(compositor)->render();
}

} // NS DX11Renderer

#endif // _DX11RENDERER_COMPOSITOR_H_