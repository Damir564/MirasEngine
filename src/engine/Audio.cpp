#include "Audio.h"
#include <SDL3/SDL_audio.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_stdinc.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "Log.h"

namespace {

// Device thread: tops the stream up with freshly mixed audio.
void SDLCALL streamCallback(void* userdata, SDL_AudioStream* stream, int additionalAmount, int /*totalAmount*/)
{
    if (additionalAmount <= 0)
        return;
    auto* audio = static_cast<AudioSystem*>(userdata);
    constexpr int kFrameBytes = static_cast<int>(sizeof(float) * 2);
    const int frames = (additionalAmount + kFrameBytes - 1) / kFrameBytes;
    // Mixed in pieces so the stack buffer stays small.
    float buffer[2 * 512];
    for (int done = 0; done < frames;) {
        const int count = std::min(frames - done, 512);
        audio->render(buffer, count);
        SDL_PutAudioStreamData(stream, buffer, count * kFrameBytes);
        done += count;
    }
}

// Linear below 0.75, then bends smoothly towards 1, so loud mixes saturate instead of clicking.
float softClip(float x)
{
    const float a = std::abs(x);
    if (a <= 0.75f)
        return x;
    const float y = 0.75f + 0.25f * std::tanh((a - 0.75f) / 0.25f);
    return x < 0.0f ? -y : y;
}

} // namespace

AudioSystem::AudioSystem() = default;

AudioSystem::~AudioSystem()
{
    shutdown();
}

bool AudioSystem::init()
{
    if (m_stream)
        return true;
    if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
        LOG_ERROR("[AUDIO] No audio: " << SDL_GetError() << "\n");
        return false;
    }
    m_subsystemStarted = true;
    const SDL_AudioSpec spec{ SDL_AUDIO_F32, 2, kSampleRate };
    m_stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, streamCallback, this);
    if (!m_stream) {
        LOG_ERROR("[AUDIO] No playback device: " << SDL_GetError() << "\n");
        return false;
    }
    SDL_ResumeAudioStreamDevice(m_stream);
    LOG_INFO("[AUDIO] Playback device open (" << kSampleRate << " Hz stereo)\n");
    return true;
}

void AudioSystem::shutdown()
{
    if (m_stream) {
        // Stops the device thread before the voices it reads go away.
        SDL_DestroyAudioStream(m_stream);
        m_stream = nullptr;
    }
    if (m_subsystemStarted) {
        SDL_QuitSubSystem(SDL_INIT_AUDIO);
        m_subsystemStarted = false;
    }
    std::lock_guard lock(m_mutex);
    m_voices.clear();
}

AudioSystem::SoundId AudioSystem::addSound(const std::string& name, std::vector<float> samples)
{
    auto shared = std::make_shared<const std::vector<float>>(std::move(samples));
    std::lock_guard lock(m_mutex);
    for (size_t i = 0; i < m_sounds.size(); ++i) {
        if (m_sounds[i].name == name) {
            m_sounds[i].samples = std::move(shared);
            return static_cast<SoundId>(i);
        }
    }
    m_sounds.push_back({ name, std::move(shared) });
    return static_cast<SoundId>(m_sounds.size() - 1);
}

AudioSystem::SoundId AudioSystem::loadSound(const std::string& name, const std::string& path)
{
    SDL_AudioSpec fileSpec{};
    Uint8* fileData = nullptr;
    Uint32 fileLength = 0;
    if (!SDL_LoadWAV(path.c_str(), &fileSpec, &fileData, &fileLength)) {
        LOG_ERROR("[AUDIO] Can't load " << path << ": " << SDL_GetError() << "\n");
        return -1;
    }
    const SDL_AudioSpec monoSpec{ SDL_AUDIO_F32, 1, kSampleRate };
    Uint8* converted = nullptr;
    int convertedLength = 0;
    const bool ok = SDL_ConvertAudioSamples(&fileSpec, fileData, static_cast<int>(fileLength), &monoSpec,
        &converted, &convertedLength);
    SDL_free(fileData);
    if (!ok) {
        LOG_ERROR("[AUDIO] Can't convert " << path << ": " << SDL_GetError() << "\n");
        return -1;
    }
    std::vector<float> samples(static_cast<size_t>(convertedLength) / sizeof(float));
    std::memcpy(samples.data(), converted, samples.size() * sizeof(float));
    SDL_free(converted);
    return addSound(name, std::move(samples));
}

AudioSystem::SoundId AudioSystem::findSound(const std::string& name) const
{
    std::lock_guard lock(m_mutex);
    for (size_t i = 0; i < m_sounds.size(); ++i)
        if (m_sounds[i].name == name)
            return static_cast<SoundId>(i);
    return -1;
}

float AudioSystem::soundDuration(SoundId sound) const
{
    std::lock_guard lock(m_mutex);
    if (sound < 0 || sound >= static_cast<SoundId>(m_sounds.size()))
        return 0.0f;
    return static_cast<float>(m_sounds[sound].samples->size()) / kSampleRate;
}

AudioSystem::VoiceId AudioSystem::play(SoundId sound, const PlayParams& params)
{
    std::lock_guard lock(m_mutex);
    if (sound < 0 || sound >= static_cast<SoundId>(m_sounds.size()) || m_sounds[sound].samples->empty())
        return 0;
    if (m_voices.size() >= kMaxVoices) {
        // Full: the one-shot that has played longest makes room.
        auto oldest = m_voices.end();
        for (auto it = m_voices.begin(); it != m_voices.end(); ++it)
            if (!it->params.loop && (oldest == m_voices.end() || it->id < oldest->id))
                oldest = it;
        if (oldest == m_voices.end())
            return 0;
        m_voices.erase(oldest);
    }
    Voice voice;
    voice.id = m_nextVoice++;
    voice.samples = m_sounds[sound].samples;
    voice.params = params;
    voice.params.pitch = std::clamp(params.pitch, 0.05f, 8.0f);
    m_voices.push_back(std::move(voice));
    return m_voices.back().id;
}

AudioSystem::VoiceId AudioSystem::play(const std::string& name, const PlayParams& params)
{
    return play(findSound(name), params);
}

void AudioSystem::stop(VoiceId voice)
{
    std::lock_guard lock(m_mutex);
    std::erase_if(m_voices, [voice](const Voice& v) { return v.id == voice; });
}

void AudioSystem::stopAll()
{
    std::lock_guard lock(m_mutex);
    m_voices.clear();
}

bool AudioSystem::isPlaying(VoiceId voice) const
{
    std::lock_guard lock(m_mutex);
    return std::any_of(m_voices.begin(), m_voices.end(), [voice](const Voice& v) { return v.id == voice; });
}

void AudioSystem::setVoicePosition(VoiceId voice, const glm::vec3& position)
{
    std::lock_guard lock(m_mutex);
    for (Voice& v : m_voices)
        if (v.id == voice)
            v.params.position = position;
}

void AudioSystem::setVoiceVolume(VoiceId voice, float volume)
{
    std::lock_guard lock(m_mutex);
    for (Voice& v : m_voices)
        if (v.id == voice)
            v.params.volume = volume;
}

void AudioSystem::setListener(const glm::vec3& position, const glm::vec3& forward, const glm::vec3& up)
{
    const glm::vec3 right = glm::cross(forward, up);
    std::lock_guard lock(m_mutex);
    m_listenerPosition = position;
    if (glm::dot(right, right) > 1e-8f)
        m_listenerRight = glm::normalize(right);
}

void AudioSystem::setVolumes(float master, float music)
{
    std::lock_guard lock(m_mutex);
    m_masterVolume = std::clamp(master, 0.0f, 1.0f);
    m_musicVolume = std::clamp(music, 0.0f, 1.0f);
}

void AudioSystem::setPaused(bool paused)
{
    std::lock_guard lock(m_mutex);
    m_paused = paused;
}

size_t AudioSystem::activeVoices() const
{
    std::lock_guard lock(m_mutex);
    return m_voices.size();
}

glm::vec2 AudioSystem::voiceGains(const Voice& voice) const
{
    const PlayParams& p = voice.params;
    float gain = p.volume * m_masterVolume * (p.group == Group::Music ? m_musicVolume : 1.0f);
    // Equal-power panning; unpositioned sounds sit in the middle.
    float pan = 0.0f;
    if (p.positional) {
        const glm::vec3 offset = p.position - m_listenerPosition;
        const float distance = glm::length(offset);
        const float minDistance = std::max(p.minDistance, 0.01f);
        const float maxDistance = std::max(p.maxDistance, minDistance + 0.01f);
        // Inverse-distance falloff that fades out completely over the last quarter of the range.
        const float falloff = minDistance / std::max(distance, minDistance);
        const float edge = std::clamp((maxDistance - distance) / (0.25f * maxDistance), 0.0f, 1.0f);
        gain *= falloff * edge;
        // Close sounds are heard by both ears.
        if (distance > 1e-3f)
            pan = glm::dot(offset / distance, m_listenerRight) * std::clamp(distance / minDistance, 0.0f, 1.0f);
    }
    const float angle = (pan + 1.0f) * 0.25f * 3.14159265f;
    return glm::vec2(std::cos(angle), std::sin(angle)) * gain;
}

void AudioSystem::render(float* out, int frames)
{
    std::fill(out, out + static_cast<size_t>(frames) * 2, 0.0f);
    std::lock_guard lock(m_mutex);
    if (m_paused)
        return;
    for (Voice& voice : m_voices) {
        const std::vector<float>& samples = *voice.samples;
        const double length = static_cast<double>(samples.size());
        const glm::vec2 gains = voiceGains(voice);
        const double step = voice.params.pitch;
        for (int f = 0; f < frames; ++f) {
            if (voice.cursor >= length) {
                if (!voice.params.loop)
                    break;
                voice.cursor = std::fmod(voice.cursor, length);
            }
            const size_t index = static_cast<size_t>(voice.cursor);
            const float t = static_cast<float>(voice.cursor - static_cast<double>(index));
            const size_t next = index + 1 < samples.size() ? index + 1 : (voice.params.loop ? 0 : index);
            const float sample = samples[index] + (samples[next] - samples[index]) * t;
            out[2 * f] += sample * gains.x;
            out[2 * f + 1] += sample * gains.y;
            voice.cursor += step;
        }
    }
    std::erase_if(m_voices, [](const Voice& v) {
        return !v.params.loop && v.cursor >= static_cast<double>(v.samples->size());
    });
    for (int i = 0; i < frames * 2; ++i)
        out[i] = softClip(out[i]);
}
