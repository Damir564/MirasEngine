#pragma once
#include <string>
#include <string_view>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "ModelTypes.h"

enum class AnimInterpolation { Linear, Step, CubicSpline };
enum class AnimPath { Translation, Rotation, Scale };

// Keyframes of one node property, as in glTF. Values are vec4: xyz for translation and scale, an xyzw
// quaternion for rotation. Cubic spline channels hold three values per key: in-tangent, value, out-tangent.
struct AnimChannel {
    int node = -1;
    AnimPath path = AnimPath::Translation;
    AnimInterpolation interpolation = AnimInterpolation::Linear;
    std::vector<float> times;
    std::vector<glm::vec4> values;
};

struct AnimClip {
    std::string name;
    float duration = 0.0f;
    std::vector<AnimChannel> channels;
};

// A node of the hierarchy with its rest pose (relative to its parent). Parents come before children.
struct AnimNode {
    std::string name;
    int parent = -1;
    glm::vec3 translation{ 0.0f };
    glm::quat rotation{ 1.0f, 0.0f, 0.0f, 0.0f };
    glm::vec3 scale{ 1.0f };
    int mesh = -1; // index into AnimatedModelData::meshes; -1 = none
};

struct NodePose {
    glm::vec3 translation{ 0.0f };
    glm::quat rotation{ 1.0f, 0.0f, 0.0f, 0.0f };
    glm::vec3 scale{ 1.0f };
};

// A glTF model kept as its node hierarchy instead of being baked into one mesh: every mesh stays in its
// own space and nodes are animated rigidly by the clips (node animation; skins are not supported).
struct AnimatedModelData {
    std::vector<AnimNode> nodes;
    std::vector<Mesh> meshes;
    std::vector<AnimClip> clips;

    int findClip(std::string_view name) const;
    int findNode(std::string_view name) const;
};

void restPose(const std::vector<AnimNode>& nodes, std::vector<NodePose>& pose);
// Overwrites the nodes the clip animates with their values at `time` (clamped to the keys).
void sampleClip(const AnimClip& clip, float time, std::vector<NodePose>& pose);
// Model-space matrix of every node from parent-relative poses.
void poseToMatrices(const std::vector<AnimNode>& nodes, const std::vector<NodePose>& pose, std::vector<glm::mat4>& out);

// Plays the clips of one model, crossfading from the previous clip when a new one starts.
class Animator {
public:
    void reset(const AnimatedModelData* model);
    // Starts the clip (-1 = rest pose) unless it is already the current one; `restart` replays it anyway.
    void play(int clip, bool loop = true, float fadeSeconds = 0.15f, float speed = 1.0f, bool restart = false);
    void play(std::string_view clipName, bool loop = true, float fadeSeconds = 0.15f, float speed = 1.0f,
        bool restart = false);
    void setSpeed(float speed) { m_speed = speed; }
    void update(float dt);

    int clip() const { return m_clip; }
    float time() const { return m_time; }
    float duration() const;
    // 0..1 through the clip (wraps for looping clips).
    float normalizedTime() const;
    // A non-looping clip reached its end.
    bool finished() const;
    // Model-space matrices of every node for the current pose.
    void evaluate(std::vector<glm::mat4>& nodeMatrices);

private:
    float clipTime(int clip, float time, bool loop) const;

    const AnimatedModelData* m_model = nullptr;
    int m_clip = -1;
    float m_time = 0.0f;
    float m_speed = 1.0f;
    bool m_loop = true;
    int m_fromClip = -1;
    float m_fromTime = 0.0f;
    bool m_fromLoop = true;
    float m_fade = 0.0f;         // seconds into the crossfade
    float m_fadeDuration = 0.0f; // 0 = no crossfade
    std::vector<NodePose> m_pose;
    std::vector<NodePose> m_fromPose;
};
