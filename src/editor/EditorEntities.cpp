#include "Editor.h"
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <glm/gtc/constants.hpp>
#include "EditorStyle.h"
#include "engine/EntityTypes.h"
#include "engine/ModelManager.h"

namespace {

constexpr ImU32 kPlaceColor = IM_COL32(255, 210, 90, 255);
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
    m_entityPlace.preset = preset >= 0 && static_cast<size_t>(preset) < info->presets.size() ? preset : -1;
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
    if (m_entityPlace.preset >= 0)
        label += std::string(" (") + info->presets[m_entityPlace.preset].label + ")";
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
    const auto index = addEntity(m_entityPlace.type, &point);
    if (!index)
        return true;
    ModelInstance& instance = m_models.getInstances()[*index];
    const EntityTypeInfo* info = findEntityType(m_entityPlace.type);
    if (info && m_entityPlace.preset >= 0) {
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
        if (type.presets.empty()) {
            button(type.label, type.id, -1, type.description);
        }
        else {
            for (size_t i = 0; i < type.presets.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                char tooltip[256];
                snprintf(tooltip, sizeof(tooltip), "%s: %s\n%s", type.label, type.presets[i].params, type.description);
                button(type.presets[i].label, type.id, static_cast<int>(i), tooltip);
                ImGui::PopID();
            }
        }
        ImGui::PopID();
    }
}
