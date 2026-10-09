#include "Editor.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <glm/gtc/constants.hpp>
#include "EditorStyle.h"
#include "engine/EntityTypes.h"
#include "engine/Lights.h"
#include "engine/ModelManager.h"

namespace {

constexpr ImU32 kPlaceColor = IM_COL32(255, 210, 90, 255);
constexpr ImU32 kLightRangeColor = IM_COL32(255, 230, 140, 110);
constexpr float kLightPlaceHeight = 2.5f;
// Floors face up more than this (normal.y); anything steeper counts as a wall.
constexpr float kFloorFacing = 0.7f;

bool isEntityMarkerModel(const GPUModel* model)
{
    if (!model)
        return false;
    for (const EntityTypeInfo& type : entityTypes())
        if (model->sourcePath == type.model)
            return true;
    return false;
}

bool validPreset(const EntityTypeInfo& type, int preset)
{
    return preset >= 0 && static_cast<size_t>(preset) < type.presets.size();
}

} // namespace

void Editor::applyEntityPreset(ModelInstance& instance, const EntityPreset& preset)
{
    std::string_view rest = preset.params;
    while (!rest.empty()) {
        const size_t end = std::min(rest.find(' '), rest.size());
        const std::string_view pair = rest.substr(0, end);
        const size_t equals = pair.find('=');
        if (equals != std::string_view::npos)
            instance.entityParams = setEntityParam(instance.entityParams, pair.substr(0, equals), pair.substr(equals + 1));
        rest = end < rest.size() ? rest.substr(end + 1) : std::string_view();
    }
    instance.color = glm::vec3(preset.color[0], preset.color[1], preset.color[2]);
}

bool Editor::entityDefaultPlaceable(const EntityTypeInfo& type)
{
    return std::all_of(type.presets.begin(), type.presets.end(), [](const EntityPreset& p) { return p.user; });
}

void Editor::drawSaveEntityPreset(const ModelInstance& instance)
{
    if (ImGui::SmallButton("Save as preset...")) {
        snprintf(m_entityPresetName, sizeof(m_entityPresetName), "%s", instance.name.c_str());
        ImGui::OpenPopup("Save entity preset");
    }
    ImGui::SetItemTooltip("Saves the current values and color as a preset of this type to %s",
        entityPresetsPath().c_str());
    if (!ImGui::BeginPopup("Save entity preset"))
        return;
    ImGui::TextUnformatted("Preset name");
    if (ImGui::IsWindowAppearing())
        ImGui::SetKeyboardFocusHere();
    const bool enter = ImGui::InputText("##presetName", m_entityPresetName, sizeof(m_entityPresetName),
        ImGuiInputTextFlags_EnterReturnsTrue);
    const std::string name = m_entityPresetName;
    const EntityTypeInfo* type = findEntityType(instance.entity);
    const bool replaces = type && std::any_of(type->presets.begin(), type->presets.end(),
        [&](const EntityPreset& p) { return p.label == name; });
    if (replaces)
        ImGui::TextColored(EditorStyle::kHighlight, "Replaces the preset with this name");
    ImGui::BeginDisabled(name.empty());
    if ((ImGui::Button("Save") || enter) && !name.empty()) {
        EntityPreset preset;
        preset.label = name;
        preset.params = instance.entityParams;
        preset.color[0] = instance.color.r;
        preset.color[1] = instance.color.g;
        preset.color[2] = instance.color.b;
        // Indices into the presets may move.
        stopEntityPlacement();
        if (saveEntityPreset(instance.entity, std::move(preset)))
            setStatus("Saved preset " + name + " to " + entityPresetsPath());
        else
            setStatus("Failed to save preset " + name, true);
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Cancel"))
        ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}

void Editor::beginEntityPlacement(const std::string& type, int preset)
{
    const EntityTypeInfo* info = findEntityType(type);
    if (!info)
        return;
    // The other tools would compete for the clicks.
    m_shapeDraw = {};
    m_faceDraw = {};
    m_levelMode = LevelEditMode::Object;
    m_entityPlace.type = type;
    m_entityPlace.preset = validPreset(*info, preset) ? preset : -1;
    setStatus("Place " + entityPlaceLabel() + ": click a floor or wall in the viewport. Esc stops");
}

void Editor::stopEntityPlacement()
{
    if (!entityPlacementActive())
        return;
    m_entityPlace = {};
    setStatus("Entity placement off");
}

std::string Editor::entityPlaceLabel() const
{
    const EntityTypeInfo* info = findEntityType(m_entityPlace.type);
    if (!info)
        return m_entityPlace.type;
    std::string label = info->label;
    if (validPreset(*info, m_entityPlace.preset))
        label += " (" + info->presets[m_entityPlace.preset].label + ")";
    return label;
}

bool Editor::entityPlacementPoint(float mouseX, float mouseY, glm::vec3& out) const
{
    const Ray ray = screenToWorldRay(mouseX, mouseY, m_sceneView.width, m_sceneView.height, getView(m_camera),
        sceneProjection());
    // Other entities' markers are not floors.
    const SubmeshHitResult hit = pickSubmesh(ray, m_models.getInstances(), [&](size_t index) {
        auto* model = m_models.getModel(index);
        return isEntityMarkerModel(model) ? nullptr : model;
    });
    glm::vec3 point;
    if (hit.hit()) {
        point = ray.origin + ray.direction * hit.t;
        const ModelInstance& instance = m_models.getInstances()[hit.instanceIndex];
        const GPUModel* model = m_models.getModel(instance.modelIndex);
        bool floor = true;
        if (model && model->polyMesh) {
            const int face = pickLevelFace(mouseX, mouseY, hit.instanceIndex);
            if (face >= 0) {
                const glm::vec3 local = model->polyMesh->faceNormal(model->polyMesh->faces[face]);
                const glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(instance.getTransformMatrix())));
                floor = glm::normalize(normalMatrix * local).y > kFloorFacing;
            }
        }
        if (!floor) {
            // A wall: stand on the floor just in front of it.
            point -= glm::normalize(ray.direction) * 0.6f;
            const float drop = distanceToSurfaceBelow(point + glm::vec3(0.0f, 0.05f, 0.0f), -1);
            point.y = std::isfinite(drop) ? point.y + 0.05f - drop : 0.0f;
        }
    }
    else {
        if (std::abs(ray.direction.y) < 1e-6f)
            return false;
        const float t = -ray.origin.y / ray.direction.y;
        if (t <= 0.0f)
            return false;
        point = ray.origin + ray.direction * t;
        point.y = 0.0f;
    }
    if (snapActive()) {
        // Half cells, so entities can stand in the middle of a floor tile as well as on its corners.
        const float step = m_gridSize * 0.5f;
        point.x = std::round(point.x / step) * step;
        point.z = std::round(point.z / step) * step;
    }
    // Hits come back a hair off the surface; floors built on the grid stay on it.
    point.y = std::round(point.y * 1000.0f) / 1000.0f;
    out = point;
    return true;
}

bool Editor::handleEntityPlaceClick(float mouseX, float mouseY)
{
    if (!entityPlacementActive())
        return false;
    glm::vec3 point;
    if (!entityPlacementPoint(mouseX, mouseY, point)) {
        setStatus("Click a surface or the ground", true);
        return true;
    }
    // Lamps hang above the floor that was clicked.
    if (m_entityPlace.type == kLightEntity)
        point.y += kLightPlaceHeight;
    const auto index = addEntity(m_entityPlace.type, &point);
    if (!index)
        return true;
    ModelInstance& instance = m_models.getInstances()[*index];
    const EntityTypeInfo* info = findEntityType(m_entityPlace.type);
    if (info && validPreset(*info, m_entityPlace.preset)) {
        const EntityPreset& preset = info->presets[m_entityPlace.preset];
        applyEntityPreset(instance, preset);
        instance.name = uniqueInstanceName(preset.label);
    }
    setStatus("Placed " + instance.name + ". Click to place another, Esc stops");
    return true;
}

void Editor::drawEntityPlaceOverlay()
{
    if (!entityPlacementActive() || m_flyMode)
        return;
    const ImVec2 origin(m_sceneView.x, m_sceneView.y);
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    ImDrawList* foreground = ImGui::GetForegroundDrawList();
    if (sceneViewContains(mouse.x, mouse.y) && !uiOwnsMouse()) {
        glm::vec3 point;
        if (entityPlacementPoint(mouse.x - origin.x, mouse.y - origin.y, point)) {
            // A ring on the floor where the entity will stand, with an arrow the way it will face.
            const glm::mat4 viewProj = sceneProjection() * getView(m_camera);
            const glm::vec3 front = getFront(m_camera);
            const float yaw = std::round(std::atan2(front.x, front.z) / glm::radians(45.0f)) * glm::radians(45.0f);
            const glm::vec3 facing(std::sin(yaw), 0.0f, std::cos(yaw));
            const auto screen = [&](const glm::vec3& p) {
                const glm::vec2 s = worldToScreen(p, viewProj, m_sceneView.width, m_sceneView.height);
                return ImVec2(origin.x + s.x, origin.y + s.y);
            };
            constexpr int kSegments = 24;
            ImVec2 ring[kSegments];
            for (int i = 0; i < kSegments; ++i) {
                const float angle = glm::two_pi<float>() * static_cast<float>(i) / kSegments;
                ring[i] = screen(point + glm::vec3(std::cos(angle), 0.0f, std::sin(angle)) * 0.45f);
            }
            foreground->AddPolyline(ring, kSegments, kPlaceColor, ImDrawFlags_Closed, 2.0f);
            foreground->AddLine(screen(point), screen(point + facing * 0.8f), kPlaceColor, 2.0f);
            foreground->AddLine(screen(point), screen(point + glm::vec3(0.0f, 1.8f, 0.0f)), kPlaceColor, 1.0f);
        }
    }
    char title[160];
    snprintf(title, sizeof(title), "Place %s: click a surface (Esc: stop)", entityPlaceLabel().c_str());
    const ImVec2 size = ImGui::CalcTextSize(title);
    const ImVec2 at(origin.x + (m_sceneView.width - size.x) * 0.5f, origin.y + 12.0f);
    foreground->AddRectFilled(ImVec2(at.x - 8.0f, at.y - 4.0f), ImVec2(at.x + size.x + 8.0f, at.y + size.y + 4.0f),
        IM_COL32(0, 0, 0, 150), 4.0f);
    foreground->AddText(at, kPlaceColor, title);
}

void Editor::drawLightOverlay()
{
    if (m_flyMode)
        return;
    const glm::mat4 viewProj = sceneProjection() * getView(m_camera);
    const ImVec2 origin(m_sceneView.x, m_sceneView.y);
    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    drawList->PushClipRect(origin, ImVec2(origin.x + m_sceneView.width, origin.y + m_sceneView.height), false);
    // worldToScreen() reports points behind the camera far off-screen; those segments are skipped.
    const auto toScreen = [&](const glm::vec3& p, ImVec2& out) {
        const glm::vec2 s = worldToScreen(p, viewProj, m_sceneView.width, m_sceneView.height);
        out = ImVec2(origin.x + s.x, origin.y + s.y);
        return s.x > -5000.0f;
    };
    const auto line = [&](const glm::vec3& a, const glm::vec3& b) {
        ImVec2 sa, sb;
        if (toScreen(a, sa) && toScreen(b, sb))
            drawList->AddLine(sa, sb, kLightRangeColor, 1.5f);
    };
    // A circle around `center` in the plane spanned by u and v (unit vectors).
    const auto circle = [&](const glm::vec3& center, const glm::vec3& u, const glm::vec3& v, float radius) {
        constexpr int kSegments = 48;
        for (int i = 0; i < kSegments; ++i) {
            const float a0 = glm::two_pi<float>() * static_cast<float>(i) / kSegments;
            const float a1 = glm::two_pi<float>() * static_cast<float>(i + 1) / kSegments;
            line(center + (u * std::cos(a0) + v * std::sin(a0)) * radius,
                center + (u * std::cos(a1) + v * std::sin(a1)) * radius);
        }
    };

    for (int index : selectedIndices()) {
        const ModelInstance& instance = m_models.getInstances()[index];
        if (instance.entity != kLightEntity)
            continue;
        const SceneLight light = lightFromInstance(instance);
        if (!light.spot) {
            // The sphere the light reaches.
            circle(light.position, { 1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, light.range);
            circle(light.position, { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, light.range);
            circle(light.position, { 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f, 0.0f }, light.range);
            continue;
        }
        // The cone, as far as the light reaches.
        const glm::vec3 axis = light.direction;
        const glm::vec3 u = glm::normalize(glm::cross(axis, std::abs(axis.y) > 0.99f ? glm::vec3(1.0f, 0.0f, 0.0f)
                                                                                      : glm::vec3(0.0f, 1.0f, 0.0f)));
        const glm::vec3 v = glm::cross(axis, u);
        const float angle = glm::radians(light.coneDegrees);
        const glm::vec3 baseCenter = light.position + axis * (light.range * std::cos(angle));
        const float baseRadius = light.range * std::sin(angle);
        circle(baseCenter, u, v, baseRadius);
        for (const glm::vec3& side : { u, -u, v, -v })
            line(light.position, baseCenter + side * baseRadius);
    }
    drawList->PopClipRect();
}

void Editor::drawEntityPalette()
{
    const float rowRight = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    bool first = true;
    const auto button = [&](const char* label, const std::string& type, int preset, const char* tooltip) {
        const float width = ImGui::CalcTextSize(label).x + ImGui::GetStyle().FramePadding.x * 2.0f;
        if (!first && ImGui::GetItemRectMax().x + ImGui::GetStyle().ItemSpacing.x + width <= rowRight)
            ImGui::SameLine();
        first = false;
        const bool active = m_entityPlace.type == type && m_entityPlace.preset == preset;
        if (active)
            ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        if (ImGui::Button(label)) {
            if (active)
                stopEntityPlacement();
            else
                beginEntityPlacement(type, preset);
        }
        if (active)
            ImGui::PopStyleColor();
        ImGui::SetItemTooltip("%s\nThen click floors or walls in the viewport to place them; Esc stops.", tooltip);
    };
    for (const EntityTypeInfo& type : entityTypes()) {
        ImGui::PushID(type.id);
        if (entityDefaultPlaceable(type))
            button(type.label, type.id, -1, type.description);
        for (size_t i = 0; i < type.presets.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            char tooltip[512];
            snprintf(tooltip, sizeof(tooltip), "%s: %s\n%s", type.label, type.presets[i].params.c_str(), type.description);
            button(type.presets[i].label.c_str(), type.id, static_cast<int>(i), tooltip);
            ImGui::PopID();
        }
        ImGui::PopID();
    }
    if (ImGui::SmallButton("Reload presets")) {
        stopEntityPlacement();
        if (loadEntityPresets())
            setStatus("Reloaded entity presets from " + entityPresetsPath());
        else
            setStatus("Can't read " + entityPresetsPath(), true);
    }
    ImGui::SetItemTooltip("Rereads %s after editing it by hand", entityPresetsPath().c_str());
}
