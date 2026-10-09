#include "GameAssets.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <utility>
#include <string>
#include <vector>
#include "engine/Audio.h"
#include "engine/Log.h"

// The game's sounds, synthesized so the game needs no audio files. A WAV file in sounds/ with the same name
// (e.g. sounds/shotgun_fire.wav) replaces the synthesized one.

namespace {

constexpr float kRate = static_cast<float>(AudioSystem::kSampleRate);
constexpr float kPi = 3.14159265f;

class Noise {
public:
    explicit Noise(uint32_t seed) : m_state(seed ? seed : 1u) {}
    // Uniform in [-1, 1).
    float next()
    {
        m_state ^= m_state << 13;
        m_state ^= m_state >> 17;
        m_state ^= m_state << 5;
        return static_cast<float>(m_state) / 2147483648.0f - 1.0f;
    }

private:
    uint32_t m_state;
};

// One-pole low-pass; cutoff in Hz.
class LowPass {
public:
    float process(float x, float cutoff)
    {
        const float a = 1.0f - std::exp(-2.0f * kPi * std::max(cutoff, 1.0f) / kRate);
        m_y += a * (x - m_y);
        return m_y;
    }

private:
    float m_y = 0.0f;
};

// Two-pole resonant band-pass (state variable filter); a formant or a ringing body.
class BandPass {
public:
    float process(float x, float frequency, float q)
    {
        const float f = 2.0f * std::sin(kPi * std::min(frequency, kRate * 0.45f) / kRate);
        m_low += f * m_band;
        const float high = x - m_low - m_band / q;
        m_band += f * high;
        return m_band;
    }

private:
    float m_low = 0.0f;
    float m_band = 0.0f;
};

std::vector<float> buffer(float seconds)
{
    return std::vector<float>(static_cast<size_t>(seconds * kRate), 0.0f);
}

float env(float t, float attack, float decay)
{
    if (t < 0.0f)
        return 0.0f;
    return (t < attack ? t / attack : 1.0f) * std::exp(-std::max(t - attack, 0.0f) / decay);
}

void normalize(std::vector<float>& samples, float peak = 0.9f)
{
    float largest = 0.0f;
    for (float s : samples)
        largest = std::max(largest, std::abs(s));
    if (largest > 1e-6f)
        for (float& s : samples)
            s *= peak / largest;
    // A short fade at the end, so the sound doesn't stop with a click.
    const size_t fade = std::min<size_t>(samples.size() / 2, 96);
    for (size_t i = 0; i < fade; ++i) {
        const float g = static_cast<float>(i) / fade;
        samples[samples.size() - 1 - i] *= g;
    }
}

// A sine sweep from f0 to f1 (exponential), phase-continuous.
struct Sweep {
    float phase = 0.0f;
    float next(float f0, float f1, float u)
    {
        const float f = f0 * std::pow(f1 / f0, std::clamp(u, 0.0f, 1.0f));
        phase += 2.0f * kPi * f / kRate;
        return std::sin(phase);
    }
};

// Band-limited-ish sawtooth (a few harmonics), for growls.
float saw(float phase, int harmonics = 8)
{
    float s = 0.0f;
    for (int k = 1; k <= harmonics; ++k)
        s += std::sin(phase * k) / k;
    return s * 0.6f;
}

void click(std::vector<float>& out, float start, float pitch, float level, uint32_t seed)
{
    Noise noise(seed);
    BandPass body;
    const size_t begin = static_cast<size_t>(start * kRate);
    const size_t length = static_cast<size_t>(0.06f * kRate);
    for (size_t i = 0; i < length && begin + i < out.size(); ++i) {
        const float t = i / kRate;
        const float n = body.process(noise.next(), 3200.0f * pitch, 4.0f) * env(t, 0.0005f, 0.006f);
        const float ring = (std::sin(2 * kPi * 1650 * pitch * t) + 0.6f * std::sin(2 * kPi * 2470 * pitch * t) +
            0.4f * std::sin(2 * kPi * 3900 * pitch * t)) * env(t, 0.0005f, 0.03f) * 0.25f;
        out[begin + i] += (n * 1.6f + ring) * level;
    }
}

std::vector<float> shotgunFire()
{
    auto out = buffer(1.1f);
    Noise noise(7);
    LowPass blast, tail;
    Sweep boom;
    for (size_t i = 0; i < out.size(); ++i) {
        const float t = i / kRate;
        const float n = noise.next();
        const float cutoff = 9000.0f * std::exp(-t / 0.05f) + 500.0f;
        const float body = blast.process(n, cutoff) * env(t, 0.001f, 0.1f) * 1.6f;
        const float crack = n * env(t, 0.0002f, 0.01f) * 0.7f;
        const float low = boom.next(120.0f, 42.0f, t / 0.25f) * env(t, 0.002f, 0.16f) * 1.1f;
        const float room = tail.process(n, 900.0f) * env(t, 0.02f, 0.38f) * 0.6f;
        out[i] = std::tanh(1.6f * (body + crack + low + room));
    }
    normalize(out);
    return out;
}

std::vector<float> shotgunPump()
{
    auto out = buffer(0.42f);
    click(out, 0.0f, 1.0f, 1.0f, 11);
    click(out, 0.2f, 1.15f, 1.1f, 12);
    // Sliding between the clicks.
    Noise noise(13);
    LowPass slide;
    for (size_t i = 0; i < out.size(); ++i) {
        const float t = i / kRate;
        const float shape = std::exp(-std::pow((t - 0.1f) / 0.05f, 2.0f)) + std::exp(-std::pow((t - 0.28f) / 0.04f, 2.0f));
        out[i] += slide.process(noise.next(), 2200.0f) * shape * 0.35f;
    }
    normalize(out, 0.8f);
    return out;
}

std::vector<float> dryClick()
{
    auto out = buffer(0.12f);
    click(out, 0.0f, 1.4f, 1.0f, 21);
    normalize(out, 0.6f);
    return out;
}

std::vector<float> shellInsert()
{
    auto out = buffer(0.3f);
    click(out, 0.0f, 0.85f, 0.8f, 31);
    click(out, 0.09f, 1.3f, 0.5f, 32);
    normalize(out, 0.7f);
    return out;
}

// Fingers in cloth: a short filtered rustle.
void rustle(std::vector<float>& out, float start, float length, float cutoff, float level, uint32_t seed)
{
    Noise noise(seed);
    LowPass soft;
    BandPass band;
    const size_t begin = static_cast<size_t>(start * kRate);
    for (size_t i = begin; i < out.size() && i < begin + static_cast<size_t>(length * kRate); ++i) {
        const float t = (i - begin) / kRate;
        const float shape = std::sin(kPi * t / length) * (0.6f + 0.4f * std::abs(noise.next()));
        out[i] += band.process(soft.process(noise.next(), cutoff), cutoff * 0.6f, 1.2f) * shape * level;
    }
}

std::vector<float> shellGrab()
{
    auto out = buffer(0.2f);
    rustle(out, 0.0f, 0.12f, 2500.0f, 1.0f, 111);
    click(out, 0.1f, 2.1f, 0.25f, 112); // the brass head touching the next shell
    normalize(out, 0.5f);
    return out;
}

std::vector<float> pocketEmpty()
{
    auto out = buffer(0.25f);
    rustle(out, 0.0f, 0.2f, 1400.0f, 1.0f, 121);
    normalize(out, 0.45f);
    return out;
}

std::vector<float> shellDrop()
{
    // A plastic hull with a brass head bouncing: a few quick, fading ticks.
    auto out = buffer(0.4f);
    click(out, 0.0f, 2.3f, 1.0f, 131);
    click(out, 0.11f, 2.5f, 0.45f, 132);
    click(out, 0.19f, 2.4f, 0.2f, 133);
    normalize(out, 0.5f);
    return out;
}

std::vector<float> pistolFire()
{
    // Sharper and shorter than the shotgun: a crack, a small boom, a short room tail.
    auto out = buffer(0.6f);
    Noise noise(141);
    LowPass blast, tail;
    Sweep boom;
    for (size_t i = 0; i < out.size(); ++i) {
        const float t = i / kRate;
        const float n = noise.next();
        const float body = blast.process(n, 12000.0f * std::exp(-t / 0.02f) + 900.0f) * env(t, 0.0005f, 0.045f) * 1.5f;
        const float crack = n * env(t, 0.0001f, 0.006f) * 0.9f;
        const float low = boom.next(180.0f, 70.0f, t / 0.12f) * env(t, 0.001f, 0.07f) * 0.7f;
        const float room = tail.process(n, 1400.0f) * env(t, 0.01f, 0.2f) * 0.4f;
        out[i] = std::tanh(1.8f * (body + crack + low + room));
    }
    normalize(out);
    return out;
}

std::vector<float> clicks(std::initializer_list<std::pair<float, float>> times, float pitch, float level, uint32_t seed,
    float length = 0.3f)
{
    auto out = buffer(length);
    uint32_t s = seed;
    for (const auto& [time, gain] : times)
        click(out, time, pitch, gain, s++);
    normalize(out, level);
    return out;
}

// Metal sliding on metal between two clicks.
std::vector<float> slideRack()
{
    auto out = buffer(0.3f);
    click(out, 0.0f, 1.3f, 0.9f, 151);
    click(out, 0.17f, 1.1f, 1.1f, 152);
    Noise noise(153);
    LowPass slide;
    for (size_t i = 0; i < out.size(); ++i) {
        const float t = i / kRate;
        const float shape = std::exp(-std::pow((t - 0.08f) / 0.04f, 2.0f));
        out[i] += slide.process(noise.next(), 3000.0f) * shape * 0.3f;
    }
    normalize(out, 0.75f);
    return out;
}

// Loose things in a pocket: many quick ticks when it's full, a few when it's nearly empty.
std::vector<float> rattle(int ticks, uint32_t seed)
{
    auto out = buffer(0.25f);
    Noise noise(seed);
    for (int i = 0; i < ticks; ++i) {
        const float time = 0.01f + 0.18f * (0.5f + 0.5f * noise.next()) * (i + 1) / ticks;
        click(out, time, 2.2f + 0.4f * noise.next(), 0.3f + 0.2f * std::abs(noise.next()), seed + 10 + i);
    }
    rustle(out, 0.0f, 0.2f, 2000.0f, 0.4f, seed + 99);
    normalize(out, 0.4f);
    return out;
}

// Something heavy hitting the floor: a thud with a metallic ring.
std::vector<float> heavyDrop(float pitch, float ringLevel, uint32_t seed)
{
    auto out = buffer(0.45f);
    Noise noise(seed);
    LowPass thud;
    Sweep body;
    for (size_t i = 0; i < out.size(); ++i) {
        const float t = i / kRate;
        out[i] = thud.process(noise.next(), 900.0f) * env(t, 0.001f, 0.03f) * 2.0f +
            body.next(140.0f * pitch, 60.0f * pitch, t / 0.1f) * env(t, 0.001f, 0.05f) * 0.6f;
    }
    click(out, 0.0f, pitch, ringLevel, seed + 1);
    click(out, 0.12f, pitch * 1.1f, ringLevel * 0.4f, seed + 2);
    normalize(out, 0.7f);
    return out;
}

// A refused action: a dull, muted knock.
std::vector<float> reject()
{
    auto out = buffer(0.18f);
    Noise noise(171);
    LowPass dull;
    Sweep knock;
    for (size_t i = 0; i < out.size(); ++i) {
        const float t = i / kRate;
        out[i] = knock.next(220.0f, 140.0f, t / 0.08f) * env(t, 0.002f, 0.035f) +
            dull.process(noise.next(), 500.0f) * env(t, 0.001f, 0.02f) * 1.2f;
    }
    normalize(out, 0.5f);
    return out;
}

std::vector<float> holster()
{
    auto out = buffer(0.3f);
    rustle(out, 0.0f, 0.25f, 1800.0f, 1.0f, 181);
    click(out, 0.2f, 0.9f, 0.2f, 182);
    normalize(out, 0.45f);
    return out;
}

// A throaty voice: sawtooth through two formants with vibrato and breath noise.
std::vector<float> growl(float seconds, float f0, float f1, float formant1, float formant2, float attack,
    float decay, uint32_t seed)
{
    auto out = buffer(seconds);
    Noise noise(seed);
    BandPass a, b, breathFilter;
    float phase = 0.0f;
    for (size_t i = 0; i < out.size(); ++i) {
        const float t = i / kRate;
        const float u = t / seconds;
        const float vibrato = 1.0f + 0.07f * std::sin(2 * kPi * 7.0f * t) + 0.04f * noise.next();
        const float f = (f0 + (f1 - f0) * u) * vibrato;
        phase += 2 * kPi * f / kRate;
        const float source = saw(phase) + 0.35f * noise.next();
        const float voice = a.process(source, formant1, 5.0f) + 0.7f * b.process(source, formant2, 6.0f);
        const float breath = breathFilter.process(noise.next(), 1500.0f, 1.5f) * 0.3f;
        out[i] = std::tanh(2.0f * (voice + breath)) * env(t, attack, decay);
    }
    normalize(out);
    return out;
}

std::vector<float> enemyDeath()
{
    auto out = growl(1.3f, 120.0f, 45.0f, 520.0f, 1100.0f, 0.03f, 0.45f, 41);
    // The body hitting the floor.
    Sweep thud;
    Noise noise(42);
    LowPass dust;
    const size_t start = static_cast<size_t>(0.85f * kRate);
    for (size_t i = start; i < out.size(); ++i) {
        const float t = (i - start) / kRate;
        out[i] += (thud.next(90.0f, 40.0f, t / 0.15f) * 0.9f + dust.process(noise.next(), 600.0f) * 1.2f) *
            env(t, 0.002f, 0.09f);
    }
    normalize(out);
    return out;
}

std::vector<float> swoosh()
{
    auto out = buffer(0.45f);
    Noise noise(51);
    BandPass band;
    for (size_t i = 0; i < out.size(); ++i) {
        const float t = i / kRate;
        const float u = t / 0.45f;
        const float shape = std::sin(kPi * std::min(u * 1.3f, 1.0f));
        out[i] = band.process(noise.next(), 400.0f + 2600.0f * u, 2.5f) * shape * shape;
    }
    normalize(out, 0.7f);
    return out;
}

std::vector<float> playerHurt()
{
    auto out = buffer(0.4f);
    Sweep thud;
    Noise noise(61);
    LowPass punch;
    for (size_t i = 0; i < out.size(); ++i) {
        const float t = i / kRate;
        out[i] = thud.next(110.0f, 45.0f, t / 0.2f) * env(t, 0.002f, 0.09f) +
            punch.process(noise.next(), 1200.0f) * env(t, 0.001f, 0.03f) * 1.5f;
    }
    auto voice = growl(0.35f, 220.0f, 160.0f, 700.0f, 1250.0f, 0.02f, 0.12f, 62);
    for (size_t i = 0; i < out.size() && i < voice.size(); ++i)
        out[i] += 0.35f * voice[i];
    normalize(out);
    return out;
}

// Notes (Hz) one after another, each a bell-like tone.
std::vector<float> chime(const std::vector<float>& notes, float step, float decay, float brightness)
{
    auto out = buffer(step * notes.size() + decay * 4.0f);
    for (size_t n = 0; n < notes.size(); ++n) {
        const size_t start = static_cast<size_t>(n * step * kRate);
        for (size_t i = start; i < out.size(); ++i) {
            const float t = (i - start) / kRate;
            const float f = notes[n];
            out[i] += (std::sin(2 * kPi * f * t) + brightness * std::sin(2 * kPi * 2 * f * t) +
                brightness * 0.5f * std::sin(2 * kPi * 3 * f * t)) * env(t, 0.003f, decay);
        }
    }
    normalize(out, 0.75f);
    return out;
}

std::vector<float> footstep()
{
    auto out = buffer(0.14f);
    Noise noise(71);
    LowPass soft;
    for (size_t i = 0; i < out.size(); ++i) {
        const float t = i / kRate;
        out[i] = soft.process(noise.next(), 900.0f) * env(t, 0.002f, 0.025f) * 2.0f +
            std::sin(2 * kPi * 70.0f * t) * env(t, 0.002f, 0.03f) * 0.6f;
    }
    normalize(out, 0.6f);
    return out;
}

std::vector<float> impact()
{
    auto out = buffer(0.12f);
    Noise noise(81);
    BandPass band;
    for (size_t i = 0; i < out.size(); ++i) {
        const float t = i / kRate;
        out[i] = band.process(noise.next(), 2600.0f, 2.0f) * env(t, 0.0005f, 0.012f) * 2.0f;
    }
    normalize(out, 0.6f);
    return out;
}

std::vector<float> fleshHit()
{
    auto out = buffer(0.15f);
    Noise noise(91);
    LowPass wet;
    for (size_t i = 0; i < out.size(); ++i) {
        const float t = i / kRate;
        out[i] = wet.process(noise.next(), 700.0f) * env(t, 0.001f, 0.03f) * 2.5f;
    }
    normalize(out, 0.7f);
    return out;
}

std::vector<float> unlock()
{
    auto out = buffer(1.4f);
    Sweep rise;
    for (size_t i = 0; i < out.size(); ++i) {
        const float t = i / kRate;
        const float tremolo = 0.75f + 0.25f * std::sin(2 * kPi * 9.0f * t);
        out[i] = rise.next(220.0f, 880.0f, t / 1.0f) * tremolo * env(t, 0.05f, 0.5f);
    }
    auto bell = chime({ 880.0f, 1318.5f }, 0.25f, 0.4f, 0.3f);
    const size_t offset = static_cast<size_t>(0.6f * kRate);
    for (size_t i = 0; i < bell.size() && offset + i < out.size(); ++i)
        out[offset + i] += bell[i] * 0.6f;
    normalize(out, 0.7f);
    return out;
}

// A seamless drone: every component repeats a whole number of times over the loop, and the noise bed
// crossfades its end into its start.
std::vector<float> ambience()
{
    constexpr float kLoop = 8.0f;
    auto out = buffer(kLoop);
    const size_t count = out.size();
    const size_t fade = static_cast<size_t>(1.0f * kRate);
    std::vector<float> bed(count + fade);
    Noise noise(101);
    LowPass a, b;
    for (float& s : bed)
        s = b.process(a.process(noise.next(), 300.0f), 300.0f) * 3.0f;
    for (size_t i = 0; i < fade; ++i) {
        const float w = static_cast<float>(i) / fade;
        bed[i] = bed[i] * w + bed[count + i] * (1.0f - w);
    }
    for (size_t i = 0; i < count; ++i) {
        const float t = i / kRate;
        const float swell = 0.6f + 0.4f * std::sin(2 * kPi * t / kLoop);
        const float drone = std::sin(2 * kPi * 55.0f * t) * 0.5f + std::sin(2 * kPi * 82.5f * t) * 0.3f * swell +
            std::sin(2 * kPi * 110.25f * t) * 0.12f * (1.0f - swell) + std::sin(2 * kPi * 41.25f * t) * 0.25f;
        out[i] = drone * 0.6f + bed[i] * 0.5f;
    }
    float largest = 0.0f;
    for (float s : out)
        largest = std::max(largest, std::abs(s));
    for (float& s : out)
        s *= 0.8f / std::max(largest, 1e-6f);
    return out;
}

} // namespace

void registerGameSounds(AudioSystem& audio)
{
    const std::pair<const char*, std::function<std::vector<float>()>> sounds[] = {
        { "shotgun_fire", shotgunFire },
        { "shotgun_pump", shotgunPump },
        { "shotgun_empty", dryClick },
        { "shell_insert", shellInsert },
        { "shell_grab", shellGrab },
        { "pocket_empty", pocketEmpty },
        { "shell_drop", shellDrop },
        { "pistol_fire", pistolFire },
        { "pistol_dry", [] { return clicks({ { 0.0f, 1.0f } }, 1.9f, 0.5f, 191, 0.1f); } },
        { "mag_eject", [] { return clicks({ { 0.0f, 0.8f }, { 0.05f, 0.5f } }, 1.2f, 0.6f, 192); } },
        { "mag_insert", [] { return clicks({ { 0.0f, 0.5f }, { 0.07f, 1.2f } }, 0.95f, 0.75f, 193); } },
        { "slide_rack", slideRack },
        { "slide_lock", [] { return clicks({ { 0.0f, 1.0f } }, 1.6f, 0.45f, 194, 0.1f); } },
        { "round_insert", [] { return clicks({ { 0.0f, 0.6f }, { 0.04f, 0.9f } }, 2.0f, 0.5f, 195, 0.15f); } },
        { "pocket_rattle_full", [] { return rattle(9, 201); } },
        { "pocket_rattle_low", [] { return rattle(2, 211); } },
        { "mag_drop", [] { return heavyDrop(1.6f, 0.5f, 221); } },
        { "gun_drop", [] { return heavyDrop(0.9f, 0.9f, 231); } },
        { "reject", reject },
        { "holster", holster },
        { "enemy_alert", [] { return growl(0.9f, 85.0f, 70.0f, 480.0f, 1150.0f, 0.08f, 0.35f, 1); } },
        { "enemy_hurt", [] { return growl(0.32f, 150.0f, 95.0f, 620.0f, 1500.0f, 0.01f, 0.12f, 2); } },
        { "enemy_death", enemyDeath },
        { "enemy_attack", swoosh },
        { "player_hurt", playerHurt },
        { "pickup_health", [] { return chime({ 523.25f, 659.25f, 783.99f, 1046.5f }, 0.07f, 0.12f, 0.3f); } },
        { "pickup_ammo", [] { return chime({ 1567.98f, 2093.0f }, 0.08f, 0.05f, 0.6f); } },
        { "footstep", footstep },
        { "impact", impact },
        { "flesh_hit", fleshHit },
        { "exit_unlock", unlock },
        { "level_complete", [] { return chime({ 392.0f, 523.25f, 659.25f, 783.99f, 1046.5f }, 0.13f, 0.35f, 0.4f); } },
        { "ambience", ambience },
    };
    int replaced = 0;
    for (const auto& [name, make] : sounds) {
        const std::string file = std::string("sounds/") + name + ".wav";
        std::error_code ec;
        if (std::filesystem::exists(file, ec) && audio.loadSound(name, file) >= 0) {
            ++replaced;
            continue;
        }
        audio.addSound(name, make());
    }
    LOG_INFO("[GAME] " << std::size(sounds) << " sounds ready (" << replaced << " from sounds/)\n");
}
