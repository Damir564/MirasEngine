#pragma once

// Constants and helpers shared by Renderer*.cpp; private to them (anonymous namespace).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <execution>
#include "Atmosphere.h"
#include "Renderer.h"

namespace {

constexpr uint8_t kVisibleMain = 1;
// A submesh casts into cascade c when bit (kVisibleShadow << c) is set.
constexpr uint8_t kVisibleShadow = 2;
constexpr size_t kCullChunkSize = 1024;
// Fewer instances than this are culled on the calling thread: handing them to the thread pool costs more.
constexpr size_t kParallelCullThreshold = 128;
// Shadow map kept while shadows are off or the Classic pipeline is active; never rendered into.
constexpr uint32_t kStubShadowMapSize = 16;
// Per light shadow view (a spotlight has one, a point light six), all redrawn every frame.
constexpr uint32_t kLightShadowMapSize = 1024;
constexpr float kLightShadowNear = 0.05f;
// Unity's selection orange; occluded parts of the outline are drawn fainter.
constexpr glm::vec3 kOutlineColor{ 1.0f, 0.4f, 0.0f };
constexpr float kOutlineOccludedAlpha = 0.4f;
constexpr float kOutlineWidthPixels = 2.0f;

// Sun irradiance above the atmosphere at intensity 1, in the units exposure 0 is calibrated for.
constexpr float kSunIrradiance = 3.2f;
constexpr float kGroundAlbedo = 0.3f;
// Height fog at fogDensity 1: a few kilometers of visibility near the ground, thinning out with height.
constexpr float kFogDensity = 0.00015f;
constexpr float kFogHeightFalloff = 1.0f / 400.0f;
// Width of a soft shadow's penumbra per meter between caster and receiver: the sun's disk, widened a
// little for the scattering of the atmosphere.
constexpr float kPenumbraPerMeter = 2.0f * kSunAngularRadius * 1.5f;
// SSAO samples for the low / medium / high settings.
constexpr int kAoSamples[] = { 0, 8, 12, 20 };

const vk::VertexInputBindingDescription2EXT kLineBinding{ 0, sizeof(GizmoVertex), vk::VertexInputRate::eVertex, 1 };
const std::array<vk::VertexInputAttributeDescription2EXT, 2> kLineAttributes = { {
    { 0, 0, vk::Format::eR32G32B32Sfloat, static_cast<uint32_t>(offsetof(GizmoVertex, position)) },
    { 1, 0, vk::Format::eR32G32B32Sfloat, static_cast<uint32_t>(offsetof(GizmoVertex, color)) },
} };

// FNV-1a, used to detect frames where a shadow cascade or the sky would come out identical.
class Hasher {
public:
    void bytes(const void* data, size_t size) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < size; ++i)
            byte(p[i]);
    }
    void byte(uint8_t value) {
        m_hash ^= value;
        m_hash *= 1099511628211ull;
    }
    uint64_t value() const { return m_hash; }

private:
    uint64_t m_hash = 1469598103934665603ull;
};

vk::SampleCountFlagBits toSampleCount(int samples)
{
    switch (samples) {
    case 8: return vk::SampleCountFlagBits::e8;
    case 4: return vk::SampleCountFlagBits::e4;
    case 2: return vk::SampleCountFlagBits::e2;
    default: return vk::SampleCountFlagBits::e1;
    }
}

// Perspective near/far recovered from a right-handed zero-to-one projection matrix.
glm::vec2 projectionDepthRange(const glm::mat4& proj)
{
    const float a = proj[2][2];
    const float b = proj[3][2];
    if (std::abs(a) < 1e-12f || std::abs(a + 1.0f) < 1e-12f)
        return { 0.1f, 1000.0f };
    return { b / a, b / (a + 1.0f) };
}

template <typename It, typename Fn>
void forEachMaybeParallel(bool parallel, It begin, It end, Fn fn)
{
    if (parallel)
        std::for_each(std::execution::par, begin, end, fn);
    else
        std::for_each(begin, end, fn);
}

} // namespace
