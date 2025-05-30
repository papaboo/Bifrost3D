// Bifrost ImGui adaptor.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_IMGUI_ADAPTOR_H_
#define _BIFROST_IMGUI_ADAPTOR_H_

#include <Bifrost/Input/Keyboard.h>
#include <Bifrost/Math/Vector.h>

#define IMGUI_DEFINE_MATH_OPERATORS
#include <ImGui/Src/imgui.h> // Convenience include of ImGui functionality

#include <functional>
#include <vector>

// ------------------------------------------------------------------------------------------------
// Forward declerations
// ------------------------------------------------------------------------------------------------
namespace Bifrost::Core { class Engine; }

namespace ImGui {

class IImGuiFrame {
public:
    virtual ~IImGuiFrame() {}
    virtual void layout_frame() = 0;
};

// ------------------------------------------------------------------------------------------------
// Dear IMGUI adaptor for Bifrost input.
// Future work:
// * Disable keyboard and mouse events based on io.WantCaptureMouse and io.WantCaptureKeyboard.
// ------------------------------------------------------------------------------------------------
class ImGuiAdaptor {
public:
    ImGuiAdaptor();

    void new_frame(const Bifrost::Core::Engine& engine);

    inline void add_frame(std::unique_ptr<IImGuiFrame> frame) { m_frames.push_back(std::move(frame)); }

    template<class ImGuiFrame, class... ArgTypes>
    inline ImGuiFrame& make_frame(ArgTypes&&... args) {
        auto unique_ptr = std::make_unique<ImGuiFrame>(std::forward<ArgTypes>(args)...);
        ImGuiFrame* ptr = unique_ptr.get();
        m_frames.push_back(std::move(unique_ptr));
        return *ptr;
    }

    inline bool is_enabled() const { return m_enabled; }
    inline void set_enabled(bool enabled) { m_enabled = enabled; }

private:
    std::vector<std::unique_ptr<IImGuiFrame>> m_frames;
    bool m_enabled;

    std::vector<std::pair<Bifrost::Input::Keyboard::Key, ImGuiKey>> m_keyboard_mapping;
};

static inline void new_frame_callback(Bifrost::Core::Engine& engine, void* adaptor) {
    static_cast<ImGuiAdaptor*>(adaptor)->new_frame(engine);
}

// ------------------------------------------------------------------------------------------------
// ImGui utility functions.
// ------------------------------------------------------------------------------------------------
inline bool InputUint(const char* label, unsigned int* v, unsigned int step = 1u, unsigned int step_fast = 100u) {
    return InputScalar(label, ImGuiDataType_U32, (void*)v, (void*)(step > 0 ? &step : NULL), (void*)(step_fast > 0 ? &step_fast : NULL));
}

struct PlotData {
    std::function<float(int)> function;
    int value_count;
    ImU32 color; // = IM_COL32(255, 255, 255, 255);
    const char* label;
};

void PlotLines(const char* label, PlotData* plots, int plot_count, const char* overlay_text, float scale_min, float scale_max, Bifrost::Math::Vector2f graph_size);
void PlotHistograms(const char* label, PlotData* plots, int plot_count, const char* overlay_text, float scale_min, float scale_max, Bifrost::Math::Vector2f graph_size);

inline void PoppedTreeNode(const char* label, std::function<void()> body) {
    if (ImGui::TreeNode(label)) {
        body();
        ImGui::TreePop();
    }
}

} // NS ImGui

#endif // _BIFROST_IMGUI_ADAPTOR_H_
