#include "Game.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include "imgui.h"

namespace {

ImU32 rgba(float r, float g, float b, float a)
{
    return ImGui::GetColorU32(ImVec4(r, g, b, a));
}

// Text with a dark shadow, readable over any background.
void shadowText(ImDrawList* draw, float size, ImVec2 pos, ImU32 color, const char* text)
{
    ImFont* font = ImGui::GetFont();
    draw->AddText(font, size, ImVec2(pos.x + 2.0f, pos.y + 2.0f), rgba(0, 0, 0, 0.6f), text);
    draw->AddText(font, size, pos, color, text);
}

ImVec2 textSize(float size, const char* text)
{
    return ImGui::GetFont()->CalcTextSizeA(size, FLT_MAX, 0.0f, text);
}

} // namespace

void Game::drawHud()
{
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImDrawList* draw = ImGui::GetBackgroundDrawList(const_cast<ImGuiViewport*>(viewport));
    const ImVec2 origin = viewport->Pos;
    const ImVec2 size = viewport->Size;
    const ImVec2 center(origin.x + size.x * 0.5f, origin.y + size.y * 0.5f);
    const float scale = std::clamp(size.y / 900.0f, 0.75f, 2.0f);
    const PlayerState& player = m_world.player();

    // Being hit: a red edge that fades.
    if (player.damageFlash > 0.0f) {
        const float a = player.damageFlash * 0.45f;
        const float band = size.y * 0.18f;
        const ImU32 red = rgba(0.8f, 0.0f, 0.0f, a);
        const ImU32 clear = rgba(0.8f, 0.0f, 0.0f, 0.0f);
        draw->AddRectFilledMultiColor(origin, ImVec2(origin.x + size.x, origin.y + band), red, red, clear, clear);
        draw->AddRectFilledMultiColor(ImVec2(origin.x, origin.y + size.y - band), ImVec2(origin.x + size.x, origin.y + size.y),
            clear, clear, red, red);
        draw->AddRectFilledMultiColor(origin, ImVec2(origin.x + band, origin.y + size.y), red, clear, clear, red);
        draw->AddRectFilledMultiColor(ImVec2(origin.x + size.x - band, origin.y), ImVec2(origin.x + size.x, origin.y + size.y),
            clear, red, red, clear);
    }
    if (!player.alive)
        return;

    // Crosshair: four ticks around a gap as wide as the pellet spread; a red X when pellets hit.
    const float gap = 9.0f * scale;
    const float tick = 7.0f * scale;
    for (int i = 0; i < 4; ++i) {
        const float dx = i == 0 ? 1.0f : i == 1 ? -1.0f : 0.0f;
        const float dy = i == 2 ? 1.0f : i == 3 ? -1.0f : 0.0f;
        const ImVec2 a(center.x + dx * gap, center.y + dy * gap);
        const ImVec2 b(center.x + dx * (gap + tick), center.y + dy * (gap + tick));
        draw->AddLine(a, b, rgba(0, 0, 0, 0.6f), 3.5f * scale);
        draw->AddLine(a, b, rgba(1, 1, 1, 0.9f), 1.6f * scale);
    }
    draw->AddCircleFilled(center, 1.6f * scale, rgba(1, 1, 1, 0.9f));
    if (player.hitMarker > 0.0f) {
        const float r0 = 5.0f * scale, r1 = 12.0f * scale;
        const ImU32 color = rgba(1.0f, 0.25f, 0.2f, player.hitMarker);
        for (int sx = -1; sx <= 1; sx += 2)
            for (int sy = -1; sy <= 1; sy += 2)
                draw->AddLine(ImVec2(center.x + sx * r0, center.y + sy * r0), ImVec2(center.x + sx * r1, center.y + sy * r1),
                    color, 2.5f * scale);
    }

    const float margin = 28.0f * scale;
    const float big = 44.0f * scale;
    const float small = 18.0f * scale;
    char text[64];

    // Health, bottom left.
    const float health = player.health;
    const float healthFraction = std::clamp(health / GameWorld::kMaxHealth, 0.0f, 1.0f);
    const bool low = health < 30.0f;
    const float pulse = low ? 0.6f + 0.4f * std::sin(static_cast<float>(ImGui::GetTime()) * 8.0f) : 1.0f;
    snprintf(text, sizeof(text), "%d", static_cast<int>(std::ceil(health)));
    const ImVec2 healthPos(origin.x + margin, origin.y + size.y - margin - big - 14.0f * scale);
    shadowText(draw, small, ImVec2(healthPos.x, healthPos.y - small), rgba(0.85f, 0.85f, 0.9f, 0.9f), "HEALTH");
    shadowText(draw, big, healthPos, low ? rgba(1.0f, 0.3f, 0.25f, pulse) : rgba(1, 1, 1, 1), text);
    const ImVec2 barMin(origin.x + margin, origin.y + size.y - margin - 8.0f * scale);
    const ImVec2 barMax(barMin.x + 220.0f * scale, barMin.y + 8.0f * scale);
    draw->AddRectFilled(barMin, barMax, rgba(0, 0, 0, 0.5f), 3.0f);
    draw->AddRectFilled(barMin, ImVec2(barMin.x + (barMax.x - barMin.x) * healthFraction, barMax.y),
        low ? rgba(0.9f, 0.2f, 0.15f, 1.0f) : rgba(0.35f, 0.85f, 0.4f, 1.0f), 3.0f);

    // Shells, bottom right: loaded as icons, the reserve as a number.
    snprintf(text, sizeof(text), "%d", player.reserve);
    const ImVec2 reserveSize = textSize(big, text);
    const ImVec2 reservePos(origin.x + size.x - margin - reserveSize.x, origin.y + size.y - margin - big - 14.0f * scale);
    shadowText(draw, big, reservePos, player.reserve == 0 && player.shells == 0 ? rgba(1, 0.3f, 0.25f, 1) : rgba(1, 1, 1, 1), text);
    const char* label = m_world.reloading() ? "RELOADING" : "SHELLS";
    shadowText(draw, small, ImVec2(origin.x + size.x - margin - textSize(small, label).x, reservePos.y - small),
        rgba(0.85f, 0.85f, 0.9f, 0.9f), label);
    const float shellWidth = 9.0f * scale, shellHeight = 22.0f * scale, shellGap = 4.0f * scale;
    float x = origin.x + size.x - margin - GameWorld::kMagazineSize * (shellWidth + shellGap);
    const float y = origin.y + size.y - margin - shellHeight;
    for (int i = 0; i < GameWorld::kMagazineSize; ++i, x += shellWidth + shellGap) {
        const bool loaded = i < player.shells;
        draw->AddRectFilled(ImVec2(x, y), ImVec2(x + shellWidth, y + shellHeight * 0.72f),
            loaded ? rgba(0.85f, 0.15f, 0.1f, 1.0f) : rgba(0.2f, 0.2f, 0.2f, 0.5f), 2.0f);
        draw->AddRectFilled(ImVec2(x, y + shellHeight * 0.72f), ImVec2(x + shellWidth, y + shellHeight),
            loaded ? rgba(0.85f, 0.65f, 0.25f, 1.0f) : rgba(0.2f, 0.2f, 0.2f, 0.5f), 2.0f);
    }

    // Enemies left and time, top center.
    const int left = m_world.enemiesTotal() - m_world.enemiesKilled();
    if (m_world.enemiesTotal() > 0) {
        if (left > 0)
            snprintf(text, sizeof(text), "Enemies left: %d", left);
        else
            snprintf(text, sizeof(text), "%s", m_world.exitUnlocked() ? "Find the exit" : "Area clear");
        const ImVec2 sz = textSize(small * 1.2f, text);
        shadowText(draw, small * 1.2f, ImVec2(center.x - sz.x * 0.5f, origin.y + margin * 0.6f),
            left > 0 ? rgba(1, 1, 1, 0.95f) : rgba(0.55f, 1.0f, 0.7f, 1.0f), text);
    }
    const int seconds = static_cast<int>(m_world.levelTime());
    snprintf(text, sizeof(text), "%d:%02d", seconds / 60, seconds % 60);
    const ImVec2 timeSize = textSize(small, text);
    shadowText(draw, small, ImVec2(origin.x + size.x - margin - timeSize.x, origin.y + margin * 0.6f),
        rgba(0.85f, 0.85f, 0.9f, 0.8f), text);

    // Messages, above the crosshair's lower half, newest at the bottom.
    float messageY = center.y + size.y * 0.18f;
    const auto& messages = m_world.messages();
    for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
        const float alpha = std::clamp(3.5f - it->age, 0.0f, 1.0f);
        const ImVec2 sz = textSize(small * 1.15f, it->text.c_str());
        shadowText(draw, small * 1.15f, ImVec2(center.x - sz.x * 0.5f, messageY), rgba(1.0f, 0.92f, 0.6f, alpha),
            it->text.c_str());
        messageY -= small * 1.5f;
    }
}
