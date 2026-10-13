#include "Game.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "imgui.h"
#include "engine/GraphicsSettings.h"

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

// Developer view of the hands, guns and storage (F3 in debug builds); the game itself shows no numbers.
void Game::drawDebugOverlay()
{
    const Arms& arms = m_world.arms();
    const ArmsState& s = arms.state();
    ImGui::SetNextWindowPos(ImVec2(ImGui::GetMainViewport()->Pos.x + 10.0f, ImGui::GetMainViewport()->Pos.y + 60.0f),
        ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.7f);
    if (ImGui::Begin("Arms (F3)", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing)) {
        const auto describe = [&](const HeldItem& item) {
            std::string text = itemName(item.kind);
            if (stackable(item.kind))
                text += " x" + std::to_string(item.count);
            if (const Magazine* magazine = s.magazine(item.magazine))
                text += " #" + std::to_string(magazine->id) + " (" + std::to_string(magazine->rounds) + "/" +
                    std::to_string(magazine->capacity) + ")";
            return text;
        };
        for (Hand hand : { Hand::Main, Hand::Off }) {
            const HandState& h = s.hand(hand);
            ImGui::Text("%s hand: %s", hand == Hand::Main ? "Main" : "Off", describe(h.item).c_str());
            if (h.busy())
                ImGui::Text("    %s %.0f%%%s", taskName(h.task), h.progress() * 100.0f, h.eventDone ? " (event done)" : "");
        }
        ImGui::Separator();
        const PistolGun& pistol = s.pistol;
        const char* where = s.handHolding(ItemKind::Pistol) == 0 ? "main hand"
            : s.handHolding(ItemKind::Pistol) == 1 ? "off hand" : s.onGround(ItemKind::Pistol) ? "ground" : "holster";
        ImGui::Text("Pistol: %s, %s", pistolStateName(pistol.state), where);
        ImGui::Text("    chambered %s, slide %s", pistol.chambered ? "yes" : "no", pistol.slideLocked ? "locked back" : "forward");
        if (const Magazine* magazine = s.magazine(pistol.magazine))
            ImGui::Text("    magazine #%d: %d/%d", magazine->id, magazine->rounds, magazine->capacity);
        else
            ImGui::Text("    no magazine");
        static constexpr const char* kChamber[] = { "empty", "loaded", "spent" };
        ImGui::Text("Shotgun: chamber %s, tube %d", kChamber[static_cast<int>(s.shotgun.chamber)], s.shotgun.tube);
        ImGui::Separator();
        for (int z = 0; z < kZoneCount; ++z) {
            const ZoneState& zone = s.zones[z];
            std::string magazines;
            for (int id : zone.magazines)
                if (const Magazine* magazine = s.magazine(id))
                    magazines += " #" + std::to_string(id) + "(" + std::to_string(magazine->rounds) + ")";
            ImGui::Text("%s: %d shells, %d rounds, mags%s", zoneName(static_cast<Zone>(z)), zone.shells, zone.rounds,
                magazines.empty() ? " -" : magazines.c_str());
        }
        ImGui::Text("Ground: %d items", static_cast<int>(s.ground.size()));
        ImGui::Separator();
        ImGui::Text("Last refused: %s", arms.lastRejection().empty() ? "-" : arms.lastRejection().c_str());
    }
    ImGui::End();
}

void Game::drawBindView(ImDrawList* draw, float scale)
{
    m_hoveredBind = -1;
    m_bindTargets.clear();
    if (m_jacket < 0.5f)
        return;
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const ImVec2 origin = viewport->Pos;
    const ImVec2 size = viewport->Size;
    if (size.x < 1.0f || size.y < 1.0f)
        return;
    m_bindCursor = glm::clamp(m_bindCursor, -glm::vec2(size.x, size.y) * 0.5f, glm::vec2(size.x, size.y) * 0.5f);
    const ImVec2 cursor(origin.x + size.x * 0.5f + m_bindCursor.x, origin.y + size.y * 0.5f + m_bindCursor.y);
    const float small = 18.0f * scale;
    const auto& binds = m_world.slotBinds();

    // The targets on screen (the same camera fillFrame() renders with); the one nearest the cursor is hovered.
    m_world.bindTargets(m_bindTargets);
    const glm::mat4 viewProj = getProjection(size.x, size.y, kCameraNearPlane, m_settings.viewDistance, m_fieldOfView) *
        getView(viewCamera());
    std::vector<ImVec2> screen(m_bindTargets.size());
    std::vector<bool> onScreen(m_bindTargets.size(), false);
    float best = 80.0f * scale;
    for (size_t i = 0; i < m_bindTargets.size(); ++i) {
        const glm::vec4 clip = viewProj * glm::vec4(m_bindTargets[i].position, 1.0f);
        if (clip.w < 0.05f)
            continue;
        const glm::vec2 ndc = glm::vec2(clip) / clip.w;
        screen[i] = ImVec2(origin.x + (ndc.x * 0.5f + 0.5f) * size.x, origin.y + (ndc.y * 0.5f + 0.5f) * size.y);
        onScreen[i] = true;
        const float distance = std::hypot(screen[i].x - cursor.x, screen[i].y - cursor.y);
        if (distance < best) {
            best = distance;
            m_hoveredBind = static_cast<int>(i);
        }
    }

    // Every target with the keys bound to it; the hovered one with its name.
    for (size_t i = 0; i < m_bindTargets.size(); ++i) {
        if (!onScreen[i])
            continue;
        const SlotBind& target = m_bindTargets[i].bind;
        std::string keys;
        for (int slot = 0; slot < kSlotCount; ++slot)
            if (binds[slot].item == target.item && (isGun(target.item) || binds[slot].zone == target.zone))
                keys += std::string(keys.empty() ? "" : " ") + slotKeyName(slot);
        const bool hovered = static_cast<int>(i) == m_hoveredBind;
        draw->AddCircle(screen[i], (hovered ? 16.0f : 9.0f) * scale, hovered ? rgba(1.0f, 0.85f, 0.3f, 1.0f) : rgba(1, 1, 1, 0.5f),
            24, (hovered ? 3.0f : 1.5f) * scale);
        if (!keys.empty())
            shadowText(draw, small, ImVec2(screen[i].x + 12.0f * scale, screen[i].y - 26.0f * scale),
                rgba(0.6f, 0.9f, 1.0f, 1.0f), keys.c_str());
        if (hovered) {
            const std::string label = slotBindName(target) + (isGun(target.item) ? "   (1 - 4: bind, left click: take)"
                                                                                  : "   (Z X C V: bind, left click: take, right click: put)");
            shadowText(draw, small, ImVec2(screen[i].x + 20.0f * scale, screen[i].y - 4.0f * scale),
                rgba(1.0f, 0.92f, 0.6f, 1.0f), label.c_str());
        }
    }
    draw->AddCircleFilled(cursor, 3.5f * scale, rgba(1, 1, 1, 0.9f));
    draw->AddCircle(cursor, 7.0f * scale, rgba(0, 0, 0, 0.6f), 16, 1.5f * scale);

    // The keys' bindings along the bottom: the gun keys, then the ammunition keys. Unbound keys are left out.
    for (int row = 0; row < 2; ++row) {
        std::string list;
        for (int slot = row * kGunSlots; slot < (row + 1) * kGunSlots && slot < kSlotCount; ++slot) {
            if (binds[slot].item == ItemKind::Empty)
                continue;
            list += std::string(list.empty() ? "" : "     ") + slotKeyName(slot) + " " + slotBindName(binds[slot]);
        }
        const ImVec2 sz = textSize(small, list.c_str());
        shadowText(draw, small, ImVec2(origin.x + (size.x - sz.x) * 0.5f, origin.y + size.y - (3.6f - 1.4f * row) * small),
            rgba(0.85f, 0.9f, 1.0f, 0.95f), list.c_str());
    }
}

void Game::drawHud()
{
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImDrawList* draw = ImGui::GetBackgroundDrawList(const_cast<ImGuiViewport*>(viewport));
    const ImVec2 origin = viewport->Pos;
    const ImVec2 size = viewport->Size;
    const ImVec2 center(origin.x + size.x * 0.5f, origin.y + size.y * 0.5f);
    const float scale = std::clamp(size.y / 900.0f, 0.75f, 2.0f);
    const PlayerState& player = m_world.player();

    // Wounds: the edges stay red, more with every wound, and pulse like a heartbeat on the last one. Being
    // hit flashes them brighter.
    const float wounded = 1.0f - std::clamp(player.health / GameWorld::kMaxHealth, 0.0f, 1.0f);
    const bool lastWound = player.alive && player.health < GameWorld::kMaxHealth * 0.4f;
    const float beat = lastWound ? 0.5f + 0.5f * std::pow(std::abs(std::sin(static_cast<float>(ImGui::GetTime()) * 2.6f)), 8.0f) : 0.0f;
    const float edge = std::max(player.damageFlash * 0.45f, wounded * 0.35f + beat * 0.25f);
    if (edge > 0.0f) {
        const float a = edge;
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
#ifndef NDEBUG
    if (m_debugOverlay)
        drawDebugOverlay();
#endif
    if (!player.alive)
        return;

    // Where the hit came from: a red arc around the middle, pointing at the attacker.
    if (player.damageFlash > 0.0f) {
        const float yaw = glm::radians(m_camera.yaw);
        const glm::vec2 forward(std::cos(yaw), std::sin(yaw));
        const glm::vec2 right(-forward.y, forward.x);
        const glm::vec3 eye = eyePosition();
        const glm::vec2 toAttacker(player.hitFrom.x - eye.x, player.hitFrom.z - eye.z);
        if (glm::length(toAttacker) > 0.01f) {
            const float angle = std::atan2(glm::dot(toAttacker, right), glm::dot(toAttacker, forward));
            const float radius = size.y * 0.22f;
            // ImGui's angles run clockwise from +X; straight ahead (angle 0) is up the screen.
            const float middle = angle - 1.5707963f;
            draw->PathArcTo(center, radius, middle - 0.35f, middle + 0.35f, 24);
            draw->PathStroke(rgba(0.95f, 0.1f, 0.05f, player.damageFlash * 0.9f), 0, 10.0f * scale);
        }
    }

    // No crosshair and no ammo counter: the gun is aimed by where its barrel points, and the shells are
    // counted on the body.
    const float margin = 28.0f * scale;
    const float big = 44.0f * scale;
    const float small = 18.0f * scale;
    char text[64];

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

    // What the pick-up key would take.
    if (const std::string prompt = m_world.interactPrompt(); !prompt.empty() && m_jacket < 0.5f) {
        const std::string line = "F  " + prompt;
        const ImVec2 sz = textSize(small, line.c_str());
        shadowText(draw, small, ImVec2(center.x - sz.x * 0.5f, center.y + size.y * 0.24f), rgba(1, 1, 1, 0.9f), line.c_str());
    }
    drawBindView(draw, scale);

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
