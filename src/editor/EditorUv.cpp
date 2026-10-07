#include "Editor.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include "EditorStyle.h"
#include "engine/ModelManager.h"

namespace {

constexpr ImU32 kUvActive = IM_COL32(255, 160, 40, 255);
constexpr ImU32 kUvOther = IM_COL32(255, 200, 120, 170);
constexpr ImU32 kUvTileLine = IM_COL32(255, 255, 255, 60);
constexpr ImU32 kUvOriginLine = IM_COL32(255, 255, 255, 140);

ImTextureID toImTexture(vk::DescriptorSet set)
{
    return static_cast<ImTextureID>(reinterpret_cast<uintptr_t>(static_cast<VkDescriptorSet>(set)));
}

float wrapDegrees(float degrees)
{
    return std::fmod(std::fmod(degrees + 180.0f, 360.0f) + 360.0f, 360.0f) - 180.0f;
}

glm::vec2 average(const std::vector<glm::vec2>& points)
{
    glm::vec2 sum(0.0f);
    for (const glm::vec2& p : points)
        sum += p;
    return points.empty() ? sum : sum / static_cast<float>(points.size());
}

} // namespace

void Editor::handleFaceKeys()
{
    if (m_levelMode != LevelEditMode::Face || !selectedLevelFace())
        return;
    if (m_keymap.pressed(EditorAction::CopyFaceAttributes))
        copyFaceAttributes();
    if (m_keymap.pressed(EditorAction::PasteFaceAttributes))
        pasteFaceAttributes();
    // One cell of the shape's grid per press.
    const float grid = selectedLevelMesh()->gridSize;
    if (m_keymap.pressed(EditorAction::FacePushOut))
        moveSelectedFaces(grid, false);
    if (m_keymap.pressed(EditorAction::FacePullIn))
        moveSelectedFaces(-grid, false);
    if (m_keymap.pressed(EditorAction::FaceExtrude))
        moveSelectedFaces(grid, true);
    if (m_keymap.pressed(EditorAction::FaceExtrudeIn))
        moveSelectedFaces(-grid, true);
    if (m_keymap.pressed(EditorAction::SelectSurface))
        selectSurface();

    // Nudging only while the mouse is over the scene, so arrow keys still work in the panels.
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    if (ImGui::GetIO().WantCaptureMouse || !sceneViewContains(mouse.x, mouse.y))
        return;
    // These actions repeat while held.
    const auto pressed = [this](EditorAction action) { return m_keymap.pressed(action); };
    glm::vec2 move(0.0f);
    float turn = 0.0f, zoom = 1.0f;
    if (pressed(EditorAction::UvLeft)) move.x += 1.0f;
    if (pressed(EditorAction::UvRight)) move.x -= 1.0f;
    if (pressed(EditorAction::UvUp)) move.y += 1.0f;
    if (pressed(EditorAction::UvDown)) move.y -= 1.0f;
    if (pressed(EditorAction::UvRotateLeft)) turn -= m_snapRotate;
    if (pressed(EditorAction::UvRotateRight)) turn += m_snapRotate;
    if (pressed(EditorAction::UvScaleUp)) zoom *= 2.0f;
    if (pressed(EditorAction::UvScaleDown)) zoom *= 0.5f;
    if (move == glm::vec2(0.0f) && turn == 0.0f && zoom == 1.0f)
        return;

    PolyMesh& mesh = *selectedLevelMesh();
    for (uint32_t f : selectedLevelFaces()) {
        PolyFace& face = mesh.faces[f];
        // One grid cell of the shape, in this face's texture units.
        const float texel = levelSlotTexelSize(mesh, face.material);
        const glm::vec2 scale = glm::max(glm::abs(face.uvScale) * texel, glm::vec2(1e-4f));
        face.uvOffset += move * (mesh.gridSize / scale);
        face.uvRotation = wrapDegrees(face.uvRotation + turn);
        face.uvScale = glm::clamp(face.uvScale * zoom, glm::vec2(1e-3f), glm::vec2(1e4f));
    }
    rebuildSelectedLevelModel(turn != 0.0f ? "rotate UVs" : zoom != 1.0f ? "scale UVs" : "move UVs");
}

void Editor::copyFaceAttributes()
{
    const PolyMesh* mesh = selectedLevelMesh();
    const PolyFace* face = selectedLevelFace();
    if (!mesh || !face)
        return;
    m_faceClipboard.valid = true;
    m_faceClipboard.hasMaterial = face->material < mesh->materials.size();
    m_faceClipboard.material = m_faceClipboard.hasMaterial ? mesh->materials[face->material] : PolyMaterial{};
    m_faceClipboard.uvScale = face->uvScale;
    m_faceClipboard.uvOffset = face->uvOffset;
    m_faceClipboard.uvRotation = face->uvRotation;
    setStatus("Copied material and UVs");
}

void Editor::pasteFaceAttributes()
{
    PolyMesh* mesh = selectedLevelMesh();
    if (!mesh || !selectedLevelFace() || !m_faceClipboard.valid)
        return;
    uint32_t slot = kNoPolyMaterial;
    if (m_faceClipboard.hasMaterial) {
        const PolyMaterial& copied = m_faceClipboard.material;
        if (!copied.materialPath.empty()) {
            slot = levelSlotFor(*mesh, copied.materialPath);
        }
        else {
            // An inline material from this or another shape: reuse an identical slot, else add one.
            const auto same = std::find_if(mesh->materials.begin(), mesh->materials.end(), [&](const PolyMaterial& m) {
                return m.materialPath.empty() && m.color == copied.color && m.roughness == copied.roughness &&
                    m.metallic == copied.metallic && m.texturePath == copied.texturePath;
            });
            if (same != mesh->materials.end()) {
                slot = static_cast<uint32_t>(same - mesh->materials.begin());
            }
            else if (mesh->materials.size() < kMaxPolyMaterialSlots) {
                slot = static_cast<uint32_t>(mesh->materials.size());
                mesh->materials.push_back(copied);
            }
        }
    }
    for (uint32_t f : selectedLevelFaces()) {
        PolyFace& face = mesh->faces[f];
        face.material = slot;
        face.uvScale = m_faceClipboard.uvScale;
        face.uvOffset = m_faceClipboard.uvOffset;
        face.uvRotation = m_faceClipboard.uvRotation;
    }
    pruneLinkedSlots(*mesh);
    rebuildSelectedLevelModel("paste UVs");
}

void Editor::wrapSelectedFaces()
{
    PolyMesh* mesh = selectedLevelMesh();
    if (!mesh || !selectedLevelFace())
        return;
    // Spreads out from the active face: each pass wraps faces next to ones already done.
    const std::vector<uint32_t> selection = selectedLevelFaces();
    std::vector<uint32_t> done{ static_cast<uint32_t>(m_selectedFace) };
    std::vector<uint32_t> pending(selection.begin() + 1, selection.end());
    for (bool progress = true; progress && !pending.empty();) {
        progress = false;
        for (auto it = pending.begin(); it != pending.end();) {
            bool wrapped = false;
            for (uint32_t from : done) {
                if (mesh->wrapFaceUVs(from, *it, levelSlotTexelSize(*mesh, mesh->faces[from].material),
                        levelSlotTexelSize(*mesh, mesh->faces[*it].material))) {
                    wrapped = true;
                    break;
                }
            }
            if (wrapped) {
                done.push_back(*it);
                it = pending.erase(it);
                progress = true;
            }
            else {
                ++it;
            }
        }
    }
    if (done.size() > 1)
        rebuildSelectedLevelModel("wrap texture");
    setStatus(pending.empty() ? "Wrapped " + std::to_string(done.size() - 1) + " faces"
        : std::to_string(pending.size()) + " selected faces aren't connected to the active face", !pending.empty());
}

void Editor::drawUvEditor(PolyMesh& mesh)
{
    const std::vector<uint32_t> selection = selectedLevelFaces();
    if (selection.empty())
        return;
    const PolyFace& active = mesh.faces[m_selectedFace];

    // Texture and tint of the active face's material.
    std::string texture;
    glm::vec4 color(1.0f);
    if (active.material < mesh.materials.size()) {
        const PolyMaterial& slot = mesh.materials[active.material];
        if (const MaterialAsset* shared = m_materials.find(slot.materialPath)) {
            texture = shared->baseColorTexture;
            color = shared->color;
        }
        else {
            texture = slot.texturePath;
            color = slot.color;
        }
    }

    const float size = std::max(std::min(ImGui::GetContentRegionAvail().x, 360.0f), 64.0f);
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const ImVec2 p1(p0.x + size, p0.y + size);
    ImGui::InvisibleButton("##uvCanvas", ImVec2(size, size), ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    const bool hovered = ImGui::IsItemHovered();
    if (hovered)
        ImGui::SetItemKeyOwner(ImGuiKey_MouseWheelY); // the wheel edits here instead of scrolling the panel

    // UV <-> canvas: the view shows m_uvViewSpan texture repeats around m_uvViewCenter, V pointing down
    // like the image.
    const glm::vec2 viewMin = m_uvViewCenter - glm::vec2(m_uvViewSpan * 0.5f);
    const auto toScreen = [&](const glm::vec2& uv) {
        const glm::vec2 t = (uv - viewMin) / m_uvViewSpan;
        return ImVec2(p0.x + t.x * size, p0.y + t.y * size);
    };

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->PushClipRect(p0, p1, true);
    drawList->AddRectFilled(p0, p1, IM_COL32(25, 25, 28, 255));
    const ImU32 tint = ImGui::ColorConvertFloat4ToU32(ImVec4(color.r, color.g, color.b, 1.0f));
    if (const vk::DescriptorSet set = m_models.previewTexture(texture))
        drawList->AddImage(ImTextureRef(toImTexture(set)), p0, p1, ImVec2(viewMin.x, viewMin.y),
            ImVec2(viewMin.x + m_uvViewSpan, viewMin.y + m_uvViewSpan), tint);
    else if (!texture.empty() || active.material < mesh.materials.size())
        drawList->AddRectFilled(p0, p1, tint);
    // Texture repeat borders.
    for (float u = std::ceil(viewMin.x); u <= viewMin.x + m_uvViewSpan; u += 1.0f)
        drawList->AddLine(toScreen({ u, viewMin.y }), toScreen({ u, viewMin.y + m_uvViewSpan }),
            u == 0.0f ? kUvOriginLine : kUvTileLine);
    for (float v = std::ceil(viewMin.y); v <= viewMin.y + m_uvViewSpan; v += 1.0f)
        drawList->AddLine(toScreen({ viewMin.x, v }), toScreen({ viewMin.x + m_uvViewSpan, v }),
            v == 0.0f ? kUvOriginLine : kUvTileLine);
    // Face outlines, the active face on top.
    std::vector<glm::vec2> uvs;
    std::vector<ImVec2> points;
    for (size_t i = selection.size(); i-- > 0;) {
        const uint32_t f = selection[i];
        mesh.faceUVs(f, levelSlotTexelSize(mesh, mesh.faces[f].material), uvs);
        points.clear();
        for (const glm::vec2& uv : uvs)
            points.push_back(toScreen(uv));
        if (points.size() >= 3)
            drawList->AddPolyline(points.data(), static_cast<int>(points.size()), i == 0 ? kUvActive : kUvOther,
                ImDrawFlags_Closed, i == 0 ? 2.5f : 1.5f);
    }
    drawList->PopClipRect();
    drawList->AddRect(p0, p1, ImGui::GetColorU32(ImGuiCol_Border));

    const ImGuiIO& io = ImGui::GetIO();
    bool changed = false;
    const char* action = "move UVs";
    // Wheel: scale the texture on the faces; Ctrl+wheel: zoom the view.
    if (hovered && io.MouseWheel != 0.0f) {
        if (io.KeyCtrl) {
            m_uvViewSpan = std::clamp(m_uvViewSpan * std::pow(0.85f, io.MouseWheel), 0.25f, 64.0f);
        }
        else {
            const float factor = std::pow(1.1f, io.MouseWheel);
            for (uint32_t f : selection)
                mesh.faces[f].uvScale = glm::clamp(mesh.faces[f].uvScale * factor, glm::vec2(1e-3f), glm::vec2(1e4f));
            changed = true;
            action = "scale UVs";
        }
    }
    // Right-drag pans the view.
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Right, 0.0f))
        m_uvViewCenter -= glm::vec2(io.MouseDelta.x, io.MouseDelta.y) / size * m_uvViewSpan;

    // Left drag: move the faces' textures, or rotate them (Shift) around each face's own center.
    if (ImGui::IsItemActivated() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        m_uvDrag = {};
        m_uvDrag.active = true;
        m_uvDrag.rotate = io.KeyShift;
        m_uvDrag.startMouse = glm::vec2(io.MousePos.x, io.MousePos.y);
        for (uint32_t f : selection) {
            m_uvDrag.startOffsets.push_back(mesh.faces[f].uvOffset);
            m_uvDrag.startRotations.push_back(mesh.faces[f].uvRotation);
        }
    }
    if (m_uvDrag.active && !ImGui::IsItemActive())
        m_uvDrag.active = false;
    if (m_uvDrag.active && m_uvDrag.startOffsets.size() == selection.size() && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        const glm::vec2 mouseDelta = glm::vec2(io.MousePos.x, io.MousePos.y) - m_uvDrag.startMouse;
        for (size_t i = 0; i < selection.size(); ++i) {
            PolyFace& face = mesh.faces[selection[i]];
            const float texel = levelSlotTexelSize(mesh, face.material);
            if (m_uvDrag.rotate) {
                float degrees = mouseDelta.x * 0.5f;
                if (snapActive() && m_snapRotate > 0.0f)
                    degrees = std::round(degrees / m_snapRotate) * m_snapRotate;
                // Keep the face where it was in texture space while it turns.
                face.uvOffset = m_uvDrag.startOffsets[i];
                face.uvRotation = m_uvDrag.startRotations[i];
                mesh.faceUVs(selection[i], texel, uvs);
                const glm::vec2 centerBefore = average(uvs);
                face.uvRotation = wrapDegrees(m_uvDrag.startRotations[i] + degrees);
                mesh.faceUVs(selection[i], texel, uvs);
                face.uvOffset += centerBefore - average(uvs);
                action = "rotate UVs";
            }
            else {
                glm::vec2 delta = mouseDelta / size * m_uvViewSpan;
                if (snapActive()) {
                    // Whole grid cells of the shape, in this face's texture units.
                    const glm::vec2 step = mesh.gridSize / glm::max(glm::abs(face.uvScale) * texel, glm::vec2(1e-4f));
                    delta = glm::round(delta / step) * step;
                }
                face.uvOffset = m_uvDrag.startOffsets[i] + delta;
            }
        }
        changed = mouseDelta != glm::vec2(0.0f);
    }

    if (ImGui::Button("Frame faces")) {
        glm::vec2 lo(FLT_MAX), hi(-FLT_MAX);
        for (uint32_t f : selection) {
            mesh.faceUVs(f, levelSlotTexelSize(mesh, mesh.faces[f].material), uvs);
            for (const glm::vec2& uv : uvs) {
                lo = glm::min(lo, uv);
                hi = glm::max(hi, uv);
            }
        }
        if (lo.x <= hi.x) {
            m_uvViewCenter = (lo + hi) * 0.5f;
            m_uvViewSpan = std::clamp(std::max(hi.x - lo.x, hi.y - lo.y) * 1.3f, 0.25f, 64.0f);
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset view")) {
        m_uvViewCenter = glm::vec2(0.5f);
        m_uvViewSpan = 3.0f;
    }
    ImGui::TextDisabled("Drag: move   Shift+drag: rotate   Wheel: scale");
    ImGui::TextDisabled("Right-drag: pan   Ctrl+wheel: zoom");

    if (changed)
        rebuildSelectedLevelModel(action);
}
