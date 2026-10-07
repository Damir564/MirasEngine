#include "Editor.h"
#include <algorithm>
#include <cfloat>
#include <cstring>
#include "EditorStyle.h"
#include "engine/EntityTypes.h"
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
            if (selectionCount() > 1) {
                ImGui::TextColored(EditorStyle::kHighlight, "%zu objects selected", selectionCount());
                ImGui::TextDisabled("Showing the active one; the gizmo moves them all.");
                ImGui::Separator();
            }
            const int index = m_gizmo.selectedInstance;
            ModelInstance& instance = m_models.getInstances()[index];
            drawInspectorHeader(instance, m_models.getModel(instance.modelIndex));
            if (!instance.group.empty())
                drawGroupSection(index);
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
                if (ImGui::ColorEdit3("Color", &current.color.x))
                    markSceneChanged();
                ImGui::SetItemTooltip("Color (tint multiplied into the model's base color)");
                ImGui::SameLine();
                if (ImGui::Button("Reset")) {
                    current.color = glm::vec3(1.0f);
                    markSceneChanged();
                }
                ImGui::SetItemTooltip("Reset the color to white");
            }
            if (hasSelection() && m_gizmo.selectedInstance == index) {
                const bool entity = !m_models.getInstances()[index].entity.empty();
                if (ImGui::CollapsingHeader("Game entity", entity ? ImGuiTreeNodeFlags_DefaultOpen : 0))
                    drawEntitySection(index);
            }
            // Level shapes pick their materials per face in the Level panel.
            if (hasSelection() && m_gizmo.selectedInstance == index) {
                const size_t modelIndex = m_models.getInstances()[index].modelIndex;
                const GPUModel* model = m_models.getModel(modelIndex);
                if (model && !model->polyMesh && ImGui::CollapsingHeader("Materials", ImGuiTreeNodeFlags_DefaultOpen))
                    drawModelMaterialOverrides(modelIndex);
            }
        }
    }
    ImGui::End();
}

void Editor::drawEntitySection(int instanceIndex)
{
    ModelInstance& instance = m_models.getInstances()[instanceIndex];
    const EntityTypeInfo* type = findEntityType(instance.entity);
    constexpr const char* kNone = "None (scenery)";
    const char* preview = instance.entity.empty() ? kNone : type ? type->label : instance.entity.c_str();
    EditorStyle::propertyLabel("Type");
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::BeginCombo("##entityType", preview)) {
        if (ImGui::Selectable(kNone, instance.entity.empty()) && !instance.entity.empty()) {
            instance.entity.clear();
            instance.entityParams.clear();
            markSceneChanged();
        }
        for (const EntityTypeInfo& option : entityTypes()) {
            if (ImGui::Selectable(option.label, instance.entity == option.id) && instance.entity != option.id) {
                instance.entity = option.id;
                instance.entityParams = option.defaultParams;
                markSceneChanged();
            }
            ImGui::SetItemTooltip("%s", option.description);
        }
        ImGui::EndCombo();
    }
    ImGui::SetItemTooltip("What the game spawns here. The object itself is hidden while playing and has no collision.");
    if (instance.entity.empty())
        return;
    if (type) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::TextWrapped("%s Faces +Z, turned by rotation Y.", type->description);
        ImGui::PopStyleColor();
    }
    // The selected objects of the same type get every change too, so a group of enemies is set up at once.
    std::vector<int> sameType;
    for (int other : selectedIndices())
        if (other != instanceIndex && m_models.getInstances()[other].entity == instance.entity)
            sameType.push_back(other);
    const auto setParam = [&](const char* key, const std::string& value) {
        instance.entityParams = setEntityParam(instance.entityParams, key, value);
        for (int other : sameType) {
            ModelInstance& o = m_models.getInstances()[other];
            o.entityParams = setEntityParam(o.entityParams, key, value);
        }
        markSceneChanged();
    };
    if (!sameType.empty())
        ImGui::TextColored(EditorStyle::kHighlight, "Changes apply to %zu selected %s objects", sameType.size() + 1,
            type ? type->label : instance.entity.c_str());

    if (type && !type->presets.empty()) {
        EditorStyle::propertyLabel("Preset");
        const float rowRight = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
        for (size_t i = 0; i < type->presets.size(); ++i) {
            const EntityPreset& preset = type->presets[i];
            const float width = ImGui::CalcTextSize(preset.label).x + ImGui::GetStyle().FramePadding.x * 2.0f;
            if (i > 0 && ImGui::GetItemRectMax().x + ImGui::GetStyle().ItemSpacing.x + width <= rowRight)
                ImGui::SameLine();
            if (ImGui::Button(preset.label)) {
                applyEntityPreset(instance, preset);
                for (int other : sameType)
                    applyEntityPreset(m_models.getInstances()[other], preset);
                markSceneChanged();
            }
            ImGui::SetItemTooltip("%s (also tints the object)", preset.params);
        }
    }

    if (type) {
        for (const EntityParamInfo& param : type->params) {
            ImGui::PushID(param.key);
            EditorStyle::propertyLabel(param.label);
            ImGui::SetNextItemWidth(-FLT_MIN);
            switch (param.kind) {
            case EntityParamInfo::Kind::Number: {
                float value = std::clamp(entityParam(instance.entityParams, param.key, param.defaultValue), param.min, param.max);
                const float speed = (param.max - param.min) / 400.0f;
                if (ImGui::DragFloat("##value", &value, speed, param.min, param.max, param.format, ImGuiSliderFlags_AlwaysClamp))
                    setParam(param.key, formatEntityNumber(value));
                break;
            }
            case EntityParamInfo::Kind::Toggle: {
                bool value = entityParam(instance.entityParams, param.key, param.defaultValue) != 0.0f;
                if (ImGui::Checkbox("##value", &value))
                    setParam(param.key, value ? "1" : "0");
                break;
            }
            case EntityParamInfo::Kind::Text: {
                char text[128];
                strncpy(text, entityParamText(instance.entityParams, param.key).c_str(), sizeof(text) - 1);
                text[sizeof(text) - 1] = '\0';
                // Values end at a space.
                if (ImGui::InputText("##value", text, sizeof(text), ImGuiInputTextFlags_CharsNoBlank))
                    setParam(param.key, text);
                break;
            }
            }
            ImGui::SetItemTooltip("%s (%s)", param.tooltip, param.key);
            ImGui::PopID();
        }
    }

    // Every value as text, for keys the widgets don't know.
    if (ImGui::TreeNode("Parameter text")) {
        char params[256];
        strncpy(params, instance.entityParams.c_str(), sizeof(params) - 1);
        params[sizeof(params) - 1] = '\0';
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::InputText("##entityParams", params, sizeof(params))) {
            instance.entityParams = params;
            markSceneChanged();
        }
        ImGui::SetItemTooltip("key=value pairs separated by spaces. Defaults: %s",
            type && *type->defaultParams ? type->defaultParams : "none");
        ImGui::TreePop();
    }
}

void Editor::drawInspectorHeader(ModelInstance& instance, const GPUModel* model)
{
    if (ImGui::Checkbox("##visible", &instance.visible))
        markSceneChanged();
    ImGui::SetItemTooltip("Visible");
    ImGui::SameLine();
    char nameBuf[256];
    strncpy(nameBuf, instance.name.c_str(), sizeof(nameBuf) - 1);
    nameBuf[sizeof(nameBuf) - 1] = '\0';
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::InputText("##name", nameBuf, sizeof(nameBuf))) {
        instance.name = nameBuf;
        markSceneChanged(); // recorded once typing ends
    }
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
    bool changed = EditorStyle::vec3Control("Position", &instance.position.x, 0.0f, 0.1f);
    changed |= EditorStyle::vec3Control("Rotation", &instance.rotation.x, 0.0f, 0.5f);
    if (EditorStyle::vec3Control("Scale", &instance.scale.x, 1.0f, 0.01f, 0.01f, 1000.0f)) {
        instance.scale = glm::max(instance.scale, glm::vec3(0.01f));
        changed = true;
    }
    if (changed)
        markSceneChanged(); // recorded when the drag ends

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
