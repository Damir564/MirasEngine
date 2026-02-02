#include "PCH.h"
#include "VulkanContext.h"
#include "Renderer.h"
#include "Types.h"

int main() {
    if (!SDL_Init(SDL_INIT_VIDEO)) return -1;
    SDL_Window* window = SDL_CreateWindow("Vulkan Refactored", 1280, 720, SDL_WINDOW_VULKAN);
    if (!window) return -1;

    try {
        VulkanContext ctx;
        ctx.init(window, "My Engine");
        {

            Renderer renderer(ctx, window);
            // renderer.loadModel("models/main_sponza/NewSponza_Main_glTF_003.gltf");
            // renderer.loadModel("models/school/school.obj");
            // renderer.loadModel("models/tomsk_school/tomsk_school.obj");
            // renderer.loadModel("models/tree/tree.glb");
            renderer.loadModel("models/bus_stop/Untitled.glb");

            Camera camera;
            bool running = true;
            bool mouseEnabled = true;
            SDL_SetWindowRelativeMouseMode(window, mouseEnabled);

            SDL_Event event;
            uint32_t lastTime = SDL_GetTicks();
            float time = 0.0f;

            while (running) {
                uint32_t currentTime = SDL_GetTicks();
                float dt = (currentTime - lastTime) / 1000.0f;
                lastTime = currentTime;

                while (SDL_PollEvent(&event)) {
                    if (event.type == SDL_EVENT_QUIT || (event.type == SDL_EVENT_KEY_DOWN && event.key.scancode == SDL_SCANCODE_ESCAPE))
                        running = false;

                    if (event.type == SDL_EVENT_MOUSE_MOTION && mouseEnabled) {
                        camera.yaw += event.motion.xrel * camera.sensitivity;
                        camera.pitch -= event.motion.yrel * camera.sensitivity;
                        camera.pitch = std::clamp(camera.pitch, -89.0f, 89.0f);
                    }
                }

                // Input
                const bool* keys = SDL_GetKeyboardState(nullptr);
                glm::vec3 front{
                    cos(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch)),
                    sin(glm::radians(camera.pitch)),
                    sin(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch))
                };
                front = glm::normalize(front);
                glm::vec3 right = glm::normalize(glm::cross(front, glm::vec3(0, 1, 0)));

                float speed = keys[SDL_SCANCODE_LSHIFT] ? camera.speed * 4.0f : camera.speed;
                if (keys[SDL_SCANCODE_W]) camera.position += front * speed * dt;
                if (keys[SDL_SCANCODE_S]) camera.position -= front * speed * dt;
                if (keys[SDL_SCANCODE_A]) camera.position -= right * speed * dt;
                if (keys[SDL_SCANCODE_D]) camera.position += right * speed * dt;

                renderer.render(camera, time);
                time += dt;
            }

            ctx.getDevice().waitIdle();
        }
        ctx.cleanup(); // Or let destructor handle it
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}