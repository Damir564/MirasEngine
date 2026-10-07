#include "Animation.h"
#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

int AnimatedModelData::findClip(std::string_view name) const
{
    for (size_t i = 0; i < clips.size(); ++i)
        if (clips[i].name == name)
            return static_cast<int>(i);
    return -1;
}

int AnimatedModelData::findNode(std::string_view name) const
{
    for (size_t i = 0; i < nodes.size(); ++i)
        if (nodes[i].name == name)
            return static_cast<int>(i);
    return -1;
}

void restPose(const std::vector<AnimNode>& nodes, std::vector<NodePose>& pose)
{
    pose.resize(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i)
        pose[i] = { nodes[i].translation, nodes[i].rotation, nodes[i].scale };
}

namespace {

glm::quat toQuat(const glm::vec4& v)
{
    return glm::quat(v.w, v.x, v.y, v.z);
}

glm::vec4 sampleChannel(const AnimChannel& channel, float time)
{
    const std::vector<float>& times = channel.times;
    const bool cubic = channel.interpolation == AnimInterpolation::CubicSpline;
    const auto value = [&](size_t key) { return channel.values[cubic ? key * 3 + 1 : key]; };
    if (times.size() == 1 || time <= times.front())
        return value(0);
    if (time >= times.back())
        return value(times.size() - 1);
    const size_t next = static_cast<size_t>(std::upper_bound(times.begin(), times.end(), time) - times.begin());
    const size_t key = next - 1;
    const float span = times[next] - times[key];
    const float u = span > 0.0f ? (time - times[key]) / span : 0.0f;
    const bool rotation = channel.path == AnimPath::Rotation;

    switch (channel.interpolation) {
    case AnimInterpolation::Step:
        return value(key);
    case AnimInterpolation::CubicSpline: {
        const float u2 = u * u, u3 = u2 * u;
        const glm::vec4 p0 = value(key);
        const glm::vec4 m0 = channel.values[key * 3 + 2] * span; // out-tangent of this key
        const glm::vec4 p1 = value(next);
        const glm::vec4 m1 = channel.values[next * 3] * span;    // in-tangent of the next one
        const glm::vec4 result = (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 +
            (u3 - u2) * m1;
        return rotation ? glm::normalize(result) : result;
    }
    case AnimInterpolation::Linear:
    default:
        if (rotation) {
            const glm::quat q = glm::slerp(toQuat(value(key)), toQuat(value(next)), u);
            return glm::vec4(q.x, q.y, q.z, q.w);
        }
        return glm::mix(value(key), value(next), u);
    }
}

} // namespace

void sampleClip(const AnimClip& clip, float time, std::vector<NodePose>& pose)
{
    for (const AnimChannel& channel : clip.channels) {
        if (channel.node < 0 || channel.node >= static_cast<int>(pose.size()) || channel.times.empty())
            continue;
        const size_t needed = channel.interpolation == AnimInterpolation::CubicSpline ? channel.times.size() * 3
                                                                                       : channel.times.size();
        if (channel.values.size() < needed)
            continue;
        const glm::vec4 v = sampleChannel(channel, time);
        NodePose& node = pose[channel.node];
        switch (channel.path) {
        case AnimPath::Translation: node.translation = glm::vec3(v); break;
        case AnimPath::Rotation: node.rotation = glm::normalize(toQuat(v)); break;
        case AnimPath::Scale: node.scale = glm::vec3(v); break;
        }
    }
}

void poseToMatrices(const std::vector<AnimNode>& nodes, const std::vector<NodePose>& pose, std::vector<glm::mat4>& out)
{
    out.resize(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        const NodePose& p = pose[i];
        const glm::mat4 local = glm::translate(glm::mat4(1.0f), p.translation) * glm::mat4_cast(p.rotation) *
            glm::scale(glm::mat4(1.0f), p.scale);
        const int parent = nodes[i].parent;
        out[i] = parent >= 0 && parent < static_cast<int>(i) ? out[parent] * local : local;
    }
}

void Animator::reset(const AnimatedModelData* model)
{
    m_model = model;
    m_clip = -1;
    m_fromClip = -1;
    m_time = m_fromTime = 0.0f;
    m_fade = m_fadeDuration = 0.0f;
    m_speed = 1.0f;
}

void Animator::play(int clip, bool loop, float fadeSeconds, float speed, bool restart)
{
    if (!m_model || clip >= static_cast<int>(m_model->clips.size()))
        clip = -1;
    m_speed = speed;
    if (clip == m_clip && !restart) {
        m_loop = loop;
        return;
    }
    // The pose being left fades out from where it is now.
    m_fromClip = m_clip;
    m_fromTime = m_time;
    m_fromLoop = m_loop;
    m_fadeDuration = m_clip == clip && restart ? 0.0f : std::max(fadeSeconds, 0.0f);
    m_fade = 0.0f;
    m_clip = clip;
    m_time = 0.0f;
    m_loop = loop;
}

void Animator::play(std::string_view clipName, bool loop, float fadeSeconds, float speed, bool restart)
{
    play(m_model ? m_model->findClip(clipName) : -1, loop, fadeSeconds, speed, restart);
}

void Animator::update(float dt)
{
    m_time += dt * m_speed;
    if (m_fadeDuration > 0.0f) {
        m_fade += dt;
        if (m_fade >= m_fadeDuration)
            m_fadeDuration = 0.0f;
    }
}

float Animator::duration() const
{
    return m_model && m_clip >= 0 ? m_model->clips[m_clip].duration : 0.0f;
}

float Animator::clipTime(int clip, float time, bool loop) const
{
    if (!m_model || clip < 0)
        return 0.0f;
    const float length = m_model->clips[clip].duration;
    if (length <= 0.0f)
        return 0.0f;
    return loop ? std::fmod(time, length) : std::min(time, length);
}

float Animator::normalizedTime() const
{
    const float length = duration();
    return length > 0.0f ? clipTime(m_clip, m_time, m_loop) / length : 1.0f;
}

bool Animator::finished() const
{
    return !m_loop && (m_clip < 0 || m_time >= duration());
}

void Animator::evaluate(std::vector<glm::mat4>& nodeMatrices)
{
    if (!m_model) {
        nodeMatrices.clear();
        return;
    }
    restPose(m_model->nodes, m_pose);
    if (m_clip >= 0)
        sampleClip(m_model->clips[m_clip], clipTime(m_clip, m_time, m_loop), m_pose);
    if (m_fadeDuration > 0.0f) {
        restPose(m_model->nodes, m_fromPose);
        if (m_fromClip >= 0)
            sampleClip(m_model->clips[m_fromClip], clipTime(m_fromClip, m_fromTime, m_fromLoop), m_fromPose);
        const float w = std::clamp(m_fade / m_fadeDuration, 0.0f, 1.0f);
        for (size_t i = 0; i < m_pose.size(); ++i) {
            m_pose[i].translation = glm::mix(m_fromPose[i].translation, m_pose[i].translation, w);
            m_pose[i].rotation = glm::slerp(m_fromPose[i].rotation, m_pose[i].rotation, w);
            m_pose[i].scale = glm::mix(m_fromPose[i].scale, m_pose[i].scale, w);
        }
    }
    poseToMatrices(m_model->nodes, m_pose, nodeMatrices);
}
