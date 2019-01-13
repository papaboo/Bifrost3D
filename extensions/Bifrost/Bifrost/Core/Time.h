// Bifrost container of time values.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_CORE_TIME_H_
#define _BIFROST_CORE_TIME_H_

namespace Bifrost {
namespace Core {

// ---------------------------------------------------------------------------
// Container for values related to time, such as total time, delta time and 
// smoothe delta time.
// Future work
//   * Time struct with smooth delta time as well. Smooth delta time is handled as smoothDt = lerp(dt, smoothDt, a), let a be 0.666 or setable by the user?
//     Or use the bitsquid approach. http://bitsquid.blogspot.dk/2010/10/time-step-smoothing.html.
//     Remember Lanister time deltas, all debts must be payed. Time, technical or loans.
// ---------------------------------------------------------------------------
class Time final {
public:
    Time()
        : m_total_time(0.0)
        , m_raw_delta_time(0.0f)
        , m_smooth_delta_time(0.0f)
        , m_time_scale(1.0f)
        , m_ticks(0u) {}

    inline double get_total_time() const { return m_total_time; }
    inline float get_raw_delta_time() const { return m_raw_delta_time; }
    inline float get_smooth_delta_time() const { return m_smooth_delta_time; }
    inline unsigned int get_ticks() const { return m_ticks; }

    inline float get_time_scale() const { return m_time_scale; }
    inline void set_time_scale(float s) { m_time_scale = s; }
    inline bool is_paused() const { return m_time_scale == 0.0f; }

    void tick(double delta_time) {
        m_total_time += delta_time * m_time_scale;
        m_raw_delta_time = float(delta_time);
        m_smooth_delta_time = float(delta_time * m_time_scale);
        ++m_ticks;
    }

private:
    double m_total_time;
    float m_raw_delta_time;
    float m_smooth_delta_time;
    float m_time_scale;
    unsigned int m_ticks;
};

} // NS Core
} // NS Bifrost

#endif // _BIFROST_CORE_TIME_H_
