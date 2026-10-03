#include "Editor.h"
#include <algorithm>
#include <cfloat>
#include <cstring>
#include "EditorStyle.h"
#include "engine/ModelManager.h"

void Editor::drawInspector()
{
    if (ImGui::Begin("Inspector", &m_showInspector)) {
        if (!hasSelection()) {
            ImGui::Spacing();
            ImGui::TextDisabled("Nothing selected.");
            ImGui::Spacing();
            ImGui::TextWrapped("Click an object in the viewport or in the Hierarchy to see and edit its properties here.");
        }
        else {
            const int index = m_gizmo.selectedInstance;
            ModelInstance& instance = m_models.getInstances()[index];
            drawInspectorHeader(instance, m_models.getModel(instance.modelIndex));
            if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen))
                drawTransformSection(index);
            // The transform buttons may have deleted or replaced the selection.
            if (hasSelection() && m_gizmo.selectedInstance == index &&
                ImGui::CollapsingHeader("Appearance", ImGuiTreeNodeFlags_DefaultOpen)) {
                ModelInstance& current = m_models.getInstances()[index];
                const ImGuiStyle& style = ImGui::GetStyle();
                const float extraWidth = ImGui::CalcTextSize("Reset").x + ImGui::CalcTextSize("Color").x +
                    style.FramePadding.x * 2 + style.ItemSpacing.x + style.ItemInnerSpacing.x;
                ImGui::SetNextItemWidth(std::max(ImGui::GetContentRegionAvail().x - extraWidth, 60.0f));
                ImGui::ColorEdit3("Color", &current.color.x);
                ImGui::SetItemTooltip("Color (tint multiplied into the model's base color)");
                ImGui::SameLine();
                if (ImGui::Button("Reset")) current.color = glm::vec3(1.0f);
                ImGui::SetItemTooltip("Reset the color to white");
            }
        }
    }
    ImGui::End();
}

void Editor::drawInspectorHeader(ModelInstance& instance, const GPUModel* model)
{
    ImGui::Checkbox("##visible", &instance.visible);
    ImGui::SetItemTooltip("Visible");
    ImGui::SameLine();
    char nameBuf[256];
    strncpy(nameBuf, instance.name.c_str(), sizeof(nameBuf) - 1);
    nameBuf[sizeof(nameBuf) - 1] = '\0';
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::InputText("##name", nameBuf, sizeof(nameBuf)))
        instance.name = nameBuf;
    if (model) {
        ImGui::TextDisabled("Model: %s", model->name.c_str());
        ImGui::TextDisabled("%s vertices, %s triangles",
            formatCount(model->vertexCount).c_str(), formatCount(model->indexCount / 3).c_str());
    }
    ImGui::Spacing();
}

void Editor::drawTransformSection(int instanceIndex)
{
    ModelInstance& instance = m_models.getInstances()[instanceIndex];
    EditorStyle::vec3Control("Position", &instance.position.x, 0.0f, 0.1f);
    EditorStyle::vec3Control("Rotation", &instance.rotation.x, 0.0f, 0.5f);
    if (EditorStyle::vec3Control("Scale", &instance.scale.x, 1.0f, 0.01f, 0.01f, 1000.0f))
        instance.scale = glm::max(instance.scale, glm::vec3(0.01f));

    ImGui::Spacing();
    const float buttonWidth = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x * 2) / 3.0f;
    if (ImGui::Button("Focus", ImVec2(buttonWidth, 0))) focusOnInstance(instanceIndex);
    ImGui::SameLine();
    if (ImGui::Button("Duplicate", ImVec2(buttonWidth, 0))) duplicateInstance(instanceIndex);
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, EditorStyle::kDanger);
    const bool deletePressed = ImGui::Button("Delete", ImVec2(buttonWidth, 0));
    ImGui::PopStyleColor();
    if (deletePressed) deleteInstance(instanceIndex);
}
