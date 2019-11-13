// DirectX 11 compositor.
//-------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
//-------------------------------------------------------------------------------------------------

#include <DX11Renderer/Compositor.h>
#include <DX11Renderer/CameraEffects.h>
#include <DX11Renderer/Utils.h>

#include <filesystem>

using namespace Bifrost::Assets;
using namespace Bifrost::Core;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;

namespace DX11Renderer {

using ODXGISwapChain1 = DX11Renderer::OwnedResourcePtr<IDXGISwapChain1>;

ODevice1 create_performant_device1(unsigned int create_device_flags) {
    // Find the best performing device (apparently the one with the most memory) and initialize that.
    struct WeightedAdapter {
        int index, dedicated_memory;

        inline bool operator<(WeightedAdapter rhs) const {
            return rhs.dedicated_memory < dedicated_memory;
        }
    };

    ODXGIFactory1 dxgi_factory1;
    THROW_DX11_ERROR(CreateDXGIFactory1(IID_PPV_ARGS(&dxgi_factory1)));

    IDXGIAdapter1* adapter = nullptr;
    std::vector<WeightedAdapter> sorted_adapters;
    for (int adapter_index = 0; dxgi_factory1->EnumAdapters1(adapter_index, &adapter) != DXGI_ERROR_NOT_FOUND; ++adapter_index) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);

        // Ignore software rendering adapters.
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            continue;

        WeightedAdapter e = { adapter_index, int(desc.DedicatedVideoMemory >> 20) };
        sorted_adapters.push_back(e);

        adapter->Release();
    }

    std::sort(sorted_adapters.begin(), sorted_adapters.end());

    // Then create the device and render context.
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* render_context = nullptr;
    for (WeightedAdapter a : sorted_adapters) {
        ODXGIAdapter1 adapter = nullptr;
        dxgi_factory1->EnumAdapters1(a.index, &adapter);

        D3D_FEATURE_LEVEL feature_level_requested = D3D_FEATURE_LEVEL_11_0;
        D3D_FEATURE_LEVEL feature_level;
        HRESULT hr = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, create_device_flags, &feature_level_requested, 1,
                                       D3D11_SDK_VERSION, &device, &feature_level, &render_context);

        if (SUCCEEDED(hr))
            break;
    }

    if (device == nullptr)
        return nullptr;

    ODevice1 device1;
    THROW_DX11_ERROR(device->QueryInterface(IID_PPV_ARGS(&device1)));
    return device1;
}

ODevice1 create_performant_debug_device1() { return create_performant_device1(D3D11_CREATE_DEVICE_DEBUG); }

//-------------------------------------------------------------------------------------------------
// DirectX 11 compositor implementation.
//-------------------------------------------------------------------------------------------------
class Compositor::Implementation {
private:
    const Window& m_window;
    const std::filesystem::path m_data_directory;
    std::vector<std::unique_ptr<IRenderer>> m_renderers;
    std::vector<std::unique_ptr<IGuiRenderer>> m_GUI_renderers;

    ODevice1 m_device;
    ODeviceContext1 m_render_context;
    ODXGISwapChain1 m_swap_chain;
    unsigned int m_sync_interval = 1;

    // Backbuffer members.
    Vector2ui m_backbuffer_size;
    ORenderTargetView m_swap_chain_RTV;

    // Camera effects
    double m_counter_hertz;
    std::vector<double> m_previous_effects_times;
    CameraEffects m_camera_effects;

public:
    Implementation(HWND& hwnd, const Window& window, const std::filesystem::path& data_directory)
        : m_window(window), m_data_directory(data_directory) {

        m_device = create_performant_device1();
        m_device->GetImmediateContext1(&m_render_context);

        { // Get the device's dxgi factory and create the swap chain.
            ODXGIDevice dxgi_device = nullptr;
            THROW_DX11_ERROR(m_device->QueryInterface(IID_PPV_ARGS(&dxgi_device)));

            // NOTE Can I use the original adapter for this?
            ODXGIAdapter adapter = nullptr;
            THROW_DX11_ERROR(dxgi_device->GetAdapter(&adapter));

            DXGI_ADAPTER_DESC adapter_description;
            adapter->GetDesc(&adapter_description);
            const char* readable_feature_level = m_device->GetFeatureLevel() == D3D_FEATURE_LEVEL_11_0 ? "11.0" : "11.1";
            printf("DirectX 11 compositor using device '%S' with feature level %s.\n", adapter_description.Description, readable_feature_level);

            ODXGIFactory2 dxgi_factory2 = nullptr;
            THROW_DX11_ERROR(adapter->GetParent(IID_PPV_ARGS(&dxgi_factory2)));

            // Create swap chain
            DXGI_SWAP_CHAIN_DESC1 swap_chain_desc1 = {};
            swap_chain_desc1.Width = window.get_width();
            swap_chain_desc1.Height = window.get_height();
            swap_chain_desc1.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
            swap_chain_desc1.SampleDesc.Count = 1;
            swap_chain_desc1.SampleDesc.Quality = 0;
            swap_chain_desc1.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
            swap_chain_desc1.BufferCount = 1;
            swap_chain_desc1.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

            THROW_DX11_ERROR(dxgi_factory2->CreateSwapChainForHwnd(m_device, hwnd, &swap_chain_desc1, nullptr, nullptr, &m_swap_chain));
        }

        { // Setup backbuffer.
            m_backbuffer_size = Vector2ui::zero();

            // Backbuffer is initialized on demand when the output dimensions are known.
            m_swap_chain_RTV = nullptr;
        }

        { // Setup tonemapper
            { // Setup timer.
                LARGE_INTEGER freq;
                QueryPerformanceFrequency(&freq);
                m_counter_hertz = 1.0 / freq.QuadPart;
            }

            // auto shader_folder_path = m_data_directory + std::filesystem::path("DX11Renderer\\Shaders\\");
            auto shader_folder_path = m_data_directory / "DX11Renderer" / "Shaders";
            m_camera_effects = CameraEffects(m_device, (std::wstring)shader_folder_path + L"/");
        }

        { // Initialize renderer containers. Initialize the 0'th index to null.
            m_renderers.resize(1);
            m_renderers[0] = std::unique_ptr<IRenderer>(nullptr);
            m_previous_effects_times.resize(1);
            m_previous_effects_times[0] = std::numeric_limits<double>::lowest();
            m_GUI_renderers.resize(1);
            m_GUI_renderers[0] = std::unique_ptr<IGuiRenderer>(nullptr);
        }
    }

    bool is_valid() const {
        return m_device.resource != nullptr;
    }

    std::unique_ptr<IRenderer>& add_renderer(RendererCreator renderer_creator) {
        IRenderer* renderer = renderer_creator(*m_device, m_window.get_width(), m_window.get_height(), m_data_directory);
        if (renderer == nullptr)
            return m_renderers[0];

        if (m_renderers.size() < Renderers::capacity()) {
            m_renderers.resize(Renderers::capacity());
            m_previous_effects_times.resize(Renderers::capacity());
        }

        Renderers::UID renderer_ID = renderer->get_ID();
        m_renderers[renderer_ID] = std::unique_ptr<IRenderer>(renderer);
        m_previous_effects_times[renderer_ID] = std::numeric_limits<double>::lowest(); // Ensures that the first delta_time is positive infinity, which in turn disables eye adaptation for the first frame.
        return m_renderers[renderer_ID];
    }

    std::unique_ptr<IGuiRenderer>& add_GUI_renderer(GuiRendererCreator renderer_creator) {
        IGuiRenderer* renderer = renderer_creator(m_device);
        if (renderer != nullptr) {
            m_GUI_renderers.push_back(std::unique_ptr<IGuiRenderer>(renderer));
            return m_GUI_renderers[m_GUI_renderers.size() - 1];
        }
        else
            return m_GUI_renderers[0];;
    }

    void render() {
        if (Cameras::begin() == Cameras::end())
            return;

        auto screenshot_filler = [](ID3D11Device1* device, ID3D11DeviceContext1* context, 
                                    bool is_HDR, ID3D11Texture2D* source_resource, Bifrost::Math::Rect<int> source_viewport) -> Screenshot {
            // Helpers
            typedef Vector4<half> Vector4h;
            typedef Vector4<unsigned char> Vector4uc;
            auto to_uchar = [](half v) -> unsigned char { return unsigned char(clamp(v * 255.0f + 0.5f, 0.0f, 255.0f)); };

            int width = source_viewport.width; 
            int height = source_viewport.height;
            int element_count = width * height;
            if (is_HDR) {
                auto pixels = std::vector<Vector4h>(element_count);
                Readback::texture2D(device, context, source_resource, source_viewport, pixels.begin());

                RGBA* data = new RGBA[element_count];
                for (int y = 0; y < height; ++y)
                    for (int x = 0; x < width; ++x) {
                        Vector4h& pixel = pixels[x + y * width];
                        data[x + (height - y - 1) * width] = RGBA(pixel.x, pixel.y, pixel.z, pixel.w);
                    }
                return Screenshot(width, height, Screenshot::Content::ColorHDR, PixelFormat::RGBA_Float, data);
            } else {
                Vector4uc* pixels = new Vector4uc[element_count];
                Readback::texture2D(device, context, source_resource, source_viewport, pixels);
                // Mirror around y.
                std::vector<Vector4uc> tmp; tmp.resize(width);
                int row_size = sizeof(Vector4uc) * width;
                for (int row = 0; row < height / 2; ++row) {
                    memcpy(tmp.data(), pixels + row * width, row_size);
                    memcpy(pixels + row * width, pixels + (height - row - 1) * width, row_size);
                    memcpy(pixels + (height - row - 1) * width, tmp.data(), row_size);
                }
                return Screenshot(width, height, Screenshot::Content::ColorLDR, PixelFormat::RGBA32, pixels);
            }
        };

        Vector2ui current_backbuffer_size = Vector2ui(m_window.get_width(), m_window.get_height());
        if (m_backbuffer_size != current_backbuffer_size) {
            
            { // Setup swap chain buffer.
                // https://msdn.microsoft.com/en-us/library/windows/desktop/bb205075(v=vs.85).aspx#Handling_Window_Resizing

                m_render_context->OMSetRenderTargets(0, 0, 0);
                m_swap_chain_RTV.release();

                THROW_DX11_ERROR(m_swap_chain->ResizeBuffers(0, 0, 0, DXGI_FORMAT_UNKNOWN, 0));

                OTexture2D swap_chain_buffer;
                THROW_DX11_ERROR(m_swap_chain->GetBuffer(0, IID_PPV_ARGS(&swap_chain_buffer)));
                THROW_DX11_ERROR(m_device->CreateRenderTargetView(swap_chain_buffer, nullptr, &m_swap_chain_RTV));
            }

            m_backbuffer_size = current_backbuffer_size;
        }

        // Tell all renderers to update.
        for (auto& renderer : m_renderers)
            if (renderer)
                renderer->handle_updates();

        // Render.
        for (Cameras::UID camera_ID : Cameras::get_z_sorted_IDs()) {

            Rectf viewport = Cameras::get_viewport(camera_ID);
            viewport.x *= m_window.get_width();
            viewport.width *= m_window.get_width();
            viewport.y *= m_window.get_height();
            viewport.height *= m_window.get_height();

            Renderers::UID renderer_ID = Cameras::get_renderer_ID(camera_ID);
            auto frame = m_renderers[renderer_ID]->render(camera_ID, int(viewport.width), int(viewport.height));

            auto take_screenshot = [&](Cameras::RequestedContent content_requested, unsigned int minimum_iteration_count, bool is_HDR) -> std::vector<Screenshot> {
                if (frame.iteration_count < minimum_iteration_count)
                    return std::vector<Screenshot>();

                auto images = m_renderers[renderer_ID]->request_auxiliary_buffers(camera_ID, content_requested, int(viewport.width), int(viewport.height));

                Screenshot::Content color_content = is_HDR ? Screenshot::Content::ColorHDR : Screenshot::Content::ColorLDR;
                if (content_requested.is_set(color_content)) {
                    ID3D11View* color_buffer_view = is_HDR ? (ID3D11View*)frame.frame_SRV : (ID3D11View*)m_swap_chain_RTV;
                    ID3D11Resource* sourceResource;
                    color_buffer_view->GetResource(&sourceResource);
                    auto screenshot = screenshot_filler(m_device, m_render_context, is_HDR, (ID3D11Texture2D*)sourceResource, frame.viewport);
                    if (screenshot.pixels != nullptr)
                        images.push_back(screenshot);
                }

                return images;
            };
            auto take_hdr_screenshot = [&](Cameras::RequestedContent content_requested, unsigned int minimum_iteration_count) -> std::vector<Screenshot> {
                return take_screenshot(content_requested, minimum_iteration_count, true);
            };
            Cameras::fill_screenshot(camera_ID, take_hdr_screenshot);

            // Compute delta time for camera effects.
            LARGE_INTEGER performance_count;
            QueryPerformanceCounter(&performance_count);
            double current_time = performance_count.QuadPart * m_counter_hertz;
            float delta_time = float(current_time - m_previous_effects_times[renderer_ID]);
            m_previous_effects_times[renderer_ID] = current_time;

            // Post process the images with the camera effects.
            Cameras::UID camera_ID = *Cameras::get_iterable().begin();
            auto effects_settings = Cameras::get_effects_settings(camera_ID);
            m_camera_effects.process(m_render_context, effects_settings, delta_time, frame.frame_SRV, m_swap_chain_RTV, frame.viewport, Recti(viewport));

            auto take_ldr_screenshot = [&](Cameras::RequestedContent content_requested, unsigned int minimum_iteration_count) -> std::vector<Screenshot> {
                return take_screenshot(content_requested, minimum_iteration_count, false);
            };
            Cameras::fill_screenshot(camera_ID, take_ldr_screenshot);
        }

        // Post render calls, fx for GUI
        for (int i = 1; i < m_GUI_renderers.size(); ++i)
            m_GUI_renderers[i]->render(m_render_context);

        m_swap_chain->Present(m_sync_interval, 0);
    }

    bool uses_v_sync() const { return m_sync_interval != 0; }
    void set_v_sync(bool use_v_sync) { m_sync_interval = unsigned int(use_v_sync); }
};

//----------------------------------------------------------------------------
// DirectX 11 compositor.
//----------------------------------------------------------------------------
Compositor* Compositor::initialize(HWND& hwnd, const Bifrost::Core::Window& window, const std::filesystem::path& data_directory) {
    Compositor* c = new Compositor(hwnd, window, data_directory);
    if (!c->m_impl->is_valid()) {
        delete c;
        return nullptr;
    }

    return c;
}

Compositor::Compositor(HWND& hwnd, const Bifrost::Core::Window& window, const std::filesystem::path& data_directory) {
    m_impl = new Implementation(hwnd, window, data_directory);
}

Compositor::~Compositor() {
    delete m_impl;
}

std::unique_ptr<IRenderer>& Compositor::add_renderer(RendererCreator renderer_creator) {
    return m_impl->add_renderer(renderer_creator);
}

std::unique_ptr<IGuiRenderer>& Compositor::add_GUI_renderer(GuiRendererCreator renderer_creator) {
    return m_impl->add_GUI_renderer(renderer_creator);
}

void Compositor::render() { m_impl->render(); }

bool Compositor::uses_v_sync() const { return m_impl->uses_v_sync(); }
void Compositor::set_v_sync(bool use_v_sync) { m_impl->set_v_sync(use_v_sync); }

} // NS DX11Renderer
