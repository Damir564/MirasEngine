#pragma once
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <glm/glm.hpp>

struct SDL_AudioStream;

// Software mixer on SDL's default playback device. Sounds are mono float PCM at kSampleRate, played as
// voices with a volume, a pitch and optionally a 3D position (distance falloff and stereo panning
// relative to the listener). Without an audio device everything still works, silently.
class AudioSystem {
public:
    static constexpr int kSampleRate = 48000;
    static constexpr size_t kMaxVoices = 48;
    using SoundId = int;      // -1 = none
    using VoiceId = uint64_t; // 0 = none

    enum class Group { Effects, Music };

    struct PlayParams {
        float volume = 1.0f;
        float pitch = 1.0f; // playback speed; 2 = an octave up
        bool loop = false;
        bool positional = false;
        glm::vec3 position{ 0.0f };
        float minDistance = 2.0f;  // full volume this close
        float maxDistance = 40.0f; // silent this far
        Group group = Group::Effects;
    };

    AudioSystem();
    ~AudioSystem();
    AudioSystem(const AudioSystem&) = delete;
    AudioSystem& operator=(const AudioSystem&) = delete;

    // Opens the default playback device; false (logged) when there is none.
    bool init();
    void shutdown();
    bool deviceOpen() const { return m_stream != nullptr; }

    // Adds the sound, or replaces the one with the same name (voices playing it keep the old samples).
    SoundId addSound(const std::string& name, std::vector<float> samples);
    // A WAV file converted to mono at kSampleRate; -1 (logged) when it can't be read.
    SoundId loadSound(const std::string& name, const std::string& path);
    SoundId findSound(const std::string& name) const;
    float soundDuration(SoundId sound) const;

    VoiceId play(SoundId sound, const PlayParams& params);
    VoiceId play(const std::string& name, const PlayParams& params);
    void stop(VoiceId voice);
    void stopAll();
    bool isPlaying(VoiceId voice) const;
    void setVoicePosition(VoiceId voice, const glm::vec3& position);
    void setVoiceVolume(VoiceId voice, float volume);

    void setListener(const glm::vec3& position, const glm::vec3& forward, const glm::vec3& up);
    void setVolumes(float master, float music);
    // Paused voices keep their place and resume where they were.
    void setPaused(bool paused);
    size_t activeVoices() const;

    // Mixes the next `frames` stereo frames (interleaved) and advances the voices; the device thread
    // calls this, and it can be called directly when there is no device.
    void render(float* out, int frames);

private:
    struct Sound {
        std::string name;
        std::shared_ptr<const std::vector<float>> samples;
    };
    struct Voice {
        VoiceId id = 0;
        std::shared_ptr<const std::vector<float>> samples;
        double cursor = 0.0;
        PlayParams params;
    };
    // Left/right gains of a voice for the current listener.
    glm::vec2 voiceGains(const Voice& voice) const;

    mutable std::mutex m_mutex;
    std::vector<Sound> m_sounds;
    std::vector<Voice> m_voices;
    VoiceId m_nextVoice = 1;
    glm::vec3 m_listenerPosition{ 0.0f };
    glm::vec3 m_listenerRight{ 1.0f, 0.0f, 0.0f };
    float m_masterVolume = 1.0f;
    float m_musicVolume = 1.0f;
    bool m_paused = false;

    SDL_AudioStream* m_stream = nullptr;
    bool m_subsystemStarted = false;
    std::vector<float> m_deviceBuffer; // device thread only
};
