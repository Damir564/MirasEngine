#include <iostream>
#include <vulkan/vulkan.hpp>
#include <volk.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#define VKB_DISABLE_DEBUG_BREAK
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>
#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>

struct Vertex {
	glm::vec3 position;
};

std::vector<Vertex> vertices = {
	Vertex{{0.0f, -0.5f, 0.0f}},
	Vertex{{0.5f,  0.5f, 0.0f}},
	Vertex{{-0.5f, 0.5f, 0.0f}},
};

class VertexBuffer {
public:
	VertexBuffer(VmaAllocator allocator,
		vk::Device device,
		const std::vector<Vertex>& vertices)
		: m_allocator(allocator), m_device(device), m_vertexCount(static_cast<uint32_t>(vertices.size()))
	{
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = sizeof(Vertex) * vertices.size();
		bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;  // Changed to CPU_TO_GPU
		allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT; // auto-mapped

		VkBuffer rawBuffer;
		if (vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &rawBuffer, &m_allocation, nullptr) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create vertex buffer");
		}

		m_buffer = vk::Buffer(rawBuffer);

		// Copy vertex data
		void* mapped;
		vmaMapMemory(m_allocator, m_allocation, &mapped);
		memcpy(mapped, vertices.data(), sizeof(Vertex) * vertices.size());
		vmaUnmapMemory(m_allocator, m_allocation);
	}

	~VertexBuffer() {
		cleanup();
		//if (m_buffer && m_allocation) {
		//	vmaDestroyBuffer(m_allocator, VkBuffer(m_buffer), m_allocation);
		//}
	}

	// Disable copy (unique ownership)
	VertexBuffer(const VertexBuffer&) = delete;
	VertexBuffer& operator=(const VertexBuffer&) = delete;

	// Enable move
	VertexBuffer(VertexBuffer&& other) noexcept
		: m_allocator(other.m_allocator),
		m_device(other.m_device),
		m_buffer(other.m_buffer),
		m_allocation(other.m_allocation),
		m_vertexCount(other.m_vertexCount)
	{
		other.m_buffer = nullptr;
		other.m_allocation = nullptr;
		other.m_vertexCount = 0;
	}

	VertexBuffer& operator=(VertexBuffer&& other) noexcept {
		if (this != &other) {
			cleanup();
			m_allocator = other.m_allocator;
			m_device = other.m_device;
			m_buffer = other.m_buffer;
			m_allocation = other.m_allocation;
			m_vertexCount = other.m_vertexCount;

			other.m_buffer = nullptr;
			other.m_allocation = nullptr;
			other.m_vertexCount = 0;
		}
		return *this;
	}

	vk::Buffer& getBuffer() { return m_buffer; }
	uint32_t getVertexCount() const { return m_vertexCount; }

private:
	void cleanup() {
		if (m_buffer && m_allocation) {
			vmaDestroyBuffer(m_allocator, VkBuffer(m_buffer), m_allocation);
			m_buffer = nullptr;
			m_allocation = nullptr;
		}
	}

	VmaAllocator m_allocator;
	vk::Device m_device;
	vk::Buffer m_buffer;
	VmaAllocation m_allocation;
	uint32_t m_vertexCount;
};

std::vector<uint32_t> loadSpirv(const std::filesystem::path& path)
{
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file)
		throw std::runtime_error("Failed to open SPIR-V file");

	size_t size = file.tellg();
	if (size % 4 != 0)
		throw std::runtime_error("Invalid SPIR-V size");

	std::vector<uint32_t> code(size / 4);
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), size);
	return code;
}

int main()
{
	// ------------------------
	// 1. Initialize SDL3
	// ------------------------
	if (!SDL_Init(SDL_INIT_VIDEO)) {
		const char* err = SDL_GetError();
		std::cerr << "SDL_Init failed: " << err << "\n";
		return -1;
	}

	std::cout << "Hello CMake." << std::endl;

	SDL_Window* window = SDL_CreateWindow(
		"Vulkan SDL3",
		1280, 720,
		SDL_WINDOW_VULKAN
	);

	if (!window) {
		std::cerr << "SDL_CreateWindow failed: " << SDL_GetError() << "\n";
		SDL_Quit();
		return -1;
	}

	// ------------------------
	// 2. Initialize volk
	// ------------------------
	if (volkInitialize() != VK_SUCCESS) {
		std::cerr << "volkInitialize failed\n";
		SDL_DestroyWindow(window);
		SDL_Quit();
		return -1;
	}
	vk::detail::defaultDispatchLoaderDynamic.init(vkGetInstanceProcAddr);

	// ------------------------
	// 3. Get SDL Vulkan extensions
	// ------------------------
	Uint32 count_instance_extensions;
	const char* const* instance_extensions = SDL_Vulkan_GetInstanceExtensions(&count_instance_extensions);
	if (instance_extensions == NULL) {
		std::cerr << "SDL_Vulkan_GetInstanceExtensions failed\n";
		SDL_DestroyWindow(window);
		SDL_Quit();
		return -1;
	}

	std::vector<const char*> extensions(instance_extensions, instance_extensions + count_instance_extensions);
	for (auto ext : extensions)
		std::cout << ext << "\n";

	// ------------------------
	// 4. Create Vulkan instance using vk-bootstrap
	// ------------------------
	vkb::InstanceBuilder builder;
	auto instRet = builder
		.set_app_name("SDL3 Vulkan App")
		.require_api_version(1, 3, 0)
		.set_minimum_instance_version(1, 3)
		// .use_default_debug_messenger()
		.enable_extensions(extensions)
		// .enable_extension(VK_EXT_SHADER_OBJECT_EXTENSION_NAME)
		.build();

	if (!instRet) {
		std::cerr << "Failed to create instance: " << instRet.error().message() << "\n";
		SDL_DestroyWindow(window);
		SDL_Quit();
		return -1;
	}

	vkb::Instance vkbInstance = instRet.value();
	VkInstance vkInstance = vkbInstance.instance;
	volkLoadInstance(vkInstance);
	vk::Instance instance(vkInstance);
	vk::detail::defaultDispatchLoaderDynamic.init(instance);

	// ------------------------
	// 6. Create Vulkan surface with SDL3
	// ------------------------
	VkSurfaceKHR surfaceVk;
	if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &surfaceVk)) {
		std::cerr << "Failed to create Vulkan surface\n";
		vkb::destroy_instance(vkbInstance);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return -1;
	}
	vk::SurfaceKHR surface(surfaceVk);

	// ------------------------
	// 5. Select physical device and create logical device
	// ------------------------
	VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{};
	shaderObjectFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT;
	shaderObjectFeatures.shaderObject = VK_TRUE;

	VkPhysicalDeviceVulkan13Features features13{};
	features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
	features13.synchronization2 = VK_TRUE;
	features13.dynamicRendering = VK_TRUE;
	features13.pNext = &shaderObjectFeatures;

	vkb::PhysicalDeviceSelector selector{ vkbInstance };
	auto physRet = selector
		.set_minimum_version(1, 3)
		.add_required_extension(vk::EXTShaderObjectExtensionName)
		.add_required_extension_features(shaderObjectFeatures)
		.set_required_features_13(features13)
		.set_surface(surface)
		.select();

	if (!physRet) {
		std::cerr << "Failed to select physical device: " << physRet.error().message() << "\n";
		vkb::destroy_surface(vkbInstance, surface);
		vkb::destroy_instance(vkbInstance);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return -1;
	}


	vkb::PhysicalDevice vkbPhys = physRet.value();

	if (!vkbPhys.are_extension_features_present(shaderObjectFeatures)) {
		std::cout << "no" << "\n";
	}

	vkb::DeviceBuilder deviceBuilder{ vkbPhys };
	auto deviceRet = deviceBuilder
		.build();

	if (!deviceRet) {
		std::cerr << "Failed to create device: " << deviceRet.error().message() << "\n";
		vkb::destroy_surface(vkbInstance, surface);
		vkb::destroy_instance(vkbInstance);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return -1;
	}

	if (!vkbPhys.is_extension_present(VK_EXT_SHADER_OBJECT_EXTENSION_NAME)) {
		throw std::runtime_error("Shader Object not supported");
	}
	vk::PhysicalDevice physicalDevice(vkbPhys.physical_device);

	vkb::Device vkbDevice = deviceRet.value();
	VkDevice vkDevice = vkbDevice.device;
	volkLoadDevice(vkDevice);
	vk::Device device(vkDevice);
	vk::detail::defaultDispatchLoaderDynamic.init(device);

	// ------------------------
	// 7. Setup VMA (minimal)
	// ------------------------
	VmaVulkanFunctions vulkanFunctions{};
	vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
	vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
	VmaAllocatorCreateInfo allocatorInfo{};
	allocatorInfo.physicalDevice = physicalDevice;
	allocatorInfo.device = device;
	allocatorInfo.instance = instance;
	allocatorInfo.pVulkanFunctions = &vulkanFunctions;
	allocatorInfo.vulkanApiVersion = vk::ApiVersion13;

	VmaAllocator allocator;
	if (vmaCreateAllocator(&allocatorInfo, &allocator) != VK_SUCCESS) {
		std::cerr << "Failed to create VMA allocator\n";
		vkb::destroy_device(vkbDevice);
		vkb::destroy_surface(vkbInstance, surface);
		vkb::destroy_instance(vkbInstance);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return -1;
	}

	std::cout << "SDL3 + Vulkan instance, device, and VMA initialized successfully!\n";
	// ------------------------
	// 10. Create Swapchain
	// ------------------------
	vkb::SwapchainBuilder swapchain_builder{ vkbDevice };
	auto swap_ret = swapchain_builder
		.set_desired_extent(1280, 720)
		.set_desired_format(VkSurfaceFormatKHR{
		VK_FORMAT_B8G8R8A8_SRGB,
		VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
			})
		.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
		.build();
	if (!swap_ret) {
		std::cerr << "Failed to create Swapchain\n";
		vmaDestroyAllocator(allocator);
		vkb::destroy_device(vkbDevice);
		vkb::destroy_surface(vkbInstance, surface);
		vkb::destroy_instance(vkbInstance);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return -1;
	}
	vkb::Swapchain vkbSwapchain = swap_ret.value();	
	// ------------------------
	// 11. Get Swapchain Images and ImageViews
	// ------------------------
	auto imagesRet = vkbSwapchain.get_images();
	if (!imagesRet) {
		std::cerr << "Failed to get swapchain images\n";
		vmaDestroyAllocator(allocator);
		vkb::destroy_swapchain(vkbSwapchain);
		vkb::destroy_device(vkbDevice);
		vkb::destroy_surface(vkbInstance, surface);
		vkb::destroy_instance(vkbInstance);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return -1;
	}
	std::vector<VkImage> swapchainImages = imagesRet.value();
	uint32_t swapchainImageCount = static_cast<uint32_t>(swapchainImages.size());
	const int MAX_FRAMES_IN_FLIGHT = swapchainImageCount;
	// const int MAX_FRAMES_IN_FLIGHT = 2;
	auto viewsRet = vkbSwapchain.get_image_views();
	if (!viewsRet) {
		std::cerr << "Failed to get swapchain image views\n";
		vmaDestroyAllocator(allocator);
		vkb::destroy_swapchain(vkbSwapchain);
		vkb::destroy_device(vkbDevice);
		vkb::destroy_surface(vkbInstance, surface);
		vkb::destroy_instance(vkbInstance);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return -1;
	}
	std::vector<VkImageView> swapchainImageViews = viewsRet.value();
	{
		// Queues
		vk::Queue graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
		vk::Queue presentQueue = vkbDevice.get_queue(vkb::QueueType::present).value();
		uint32_t graphicsQueueIndex = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();
		// ------------------------
		// 12. Command Pool
		// ------------------------
		vk::CommandPoolCreateInfo poolInfo{};
		poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		poolInfo.queueFamilyIndex = graphicsQueueIndex;

		vk::UniqueCommandPool commandPool = device.createCommandPoolUnique(poolInfo).value;
		// ------------------------
		// 13. Command Buffer
		// ------------------------
		vk::CommandBufferAllocateInfo allocInfo{};
		allocInfo.commandPool = commandPool.get();
		allocInfo.level = vk::CommandBufferLevel::ePrimary;
		allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

		auto allocatedCommandBuffers = device.allocateCommandBuffersUnique(allocInfo);
		std::vector<vk::UniqueCommandBuffer> commandBuffers = std::move(allocatedCommandBuffers.value);

		// ------------------------
		// 12. Semaphores and Fences
		// ------------------------
		std::vector<vk::UniqueSemaphore> imageAvailableSemaphores(MAX_FRAMES_IN_FLIGHT);
		std::vector<vk::UniqueSemaphore> renderFinishedSemaphores(MAX_FRAMES_IN_FLIGHT);
		
		vk::FenceCreateInfo fenceInfo{};
		fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;
		std::vector<vk::UniqueFence> inFlightFences(MAX_FRAMES_IN_FLIGHT);

		for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			imageAvailableSemaphores[i] = device.createSemaphoreUnique({}).value;
			renderFinishedSemaphores[i] = device.createSemaphoreUnique({}).value;
			inFlightFences[i] = device.createFenceUnique(fenceInfo).value;
		}

		// ------------------------
		// 13. Create Vertex Buffer
		// ------------------------
		std::unique_ptr<VertexBuffer> vertexBuffer;
		try {
			vertexBuffer = std::make_unique<VertexBuffer>(allocator, device, vertices);
		}
		catch (const std::exception& e) {
			std::cerr << "Failed to create vertex buffer: " << e.what() << "\n";
			vmaDestroyAllocator(allocator);
			vkb::destroy_swapchain(vkbSwapchain);
			vkb::destroy_device(vkbDevice);
			vkb::destroy_surface(vkbInstance, surface);
			vkb::destroy_instance(vkbInstance);
			SDL_DestroyWindow(window);
			SDL_Quit();
			return -1;
		}

		// ------------------------
		// 14. Load Shaders
		// ------------------------
		std::vector<uint32_t> vertCode, fragCode;
		try {
			vertCode = loadSpirv("shaders/triangle.vert.spv");
			fragCode = loadSpirv("shaders/triangle.frag.spv");

			std::cout << "Vertex SPIR-V words: " << vertCode.size() << "\n";
			std::cout << "Fragment SPIR-V words: " << fragCode.size() << "\n";
		} catch (const std::exception& e) {
			std::cerr << "Failed to load shaders: " << e.what() << "\n";
			vmaDestroyAllocator(allocator);
			vkb::destroy_swapchain(vkbSwapchain);
			vkb::destroy_device(vkbDevice);
			vkb::destroy_surface(vkbInstance, surface);
			vkb::destroy_instance(vkbInstance);
			SDL_DestroyWindow(window);
			SDL_Quit();
			return -1;
		}
		// ------------------------
		// 16. Create Pipeline Layout
		// ------------------------
		vk::PushConstantRange pcRange{};
		pcRange.stageFlags = vk::ShaderStageFlagBits::eVertex;
		pcRange.offset = 0;
		pcRange.size = sizeof(glm::vec2);



		// shader create info
		vk::ShaderCreateInfoEXT vertInfo{};
		vertInfo.setStage(vk::ShaderStageFlagBits::eVertex)
			.setNextStage(vk::ShaderStageFlagBits::eFragment)
			.setFlags(vk::ShaderCreateFlagBitsEXT::eLinkStage)
			.setCodeType(vk::ShaderCodeTypeEXT::eSpirv)
			.setCodeSize(vertCode.size() * sizeof(vertCode.front()))
			.setPCode(vertCode.data())
			.setPName("main")
			.setPushConstantRangeCount(1)
			.setPPushConstantRanges(&pcRange);

		vk::ShaderCreateInfoEXT fragInfo{};
		fragInfo.setStage(vk::ShaderStageFlagBits::eFragment)
			.setFlags(vk::ShaderCreateFlagBitsEXT::eLinkStage)
			.setCodeType(vk::ShaderCodeTypeEXT::eSpirv)
			.setCodeSize(fragCode.size() * sizeof(uint32_t))
			.setPCode(fragCode.data())
			.setPName("main")
			.setPushConstantRangeCount(1)
			.setPPushConstantRanges(&pcRange);

		vk::ShaderEXT vertShader, fragShader;
		vertShader = device.createShaderEXT(vertInfo).value;
		fragShader = device.createShaderEXT(fragInfo).value;


		vk::PipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.setPushConstantRanges(pcRange);
		// arrays for binding shaders
		vk::PipelineLayout pipelineLayout;
		try {
			pipelineLayout = device.createPipelineLayout(layoutInfo).value;
		}
		catch (const std::exception& e) {
			std::cerr << "Failed to create pipeline layout: " << e.what() << "\n";
			device.destroyShaderEXT(vertShader);
			device.destroyShaderEXT(fragShader);
			vmaDestroyAllocator(allocator);
			vkb::destroy_swapchain(vkbSwapchain);
			vkb::destroy_device(vkbDevice);
			vkb::destroy_surface(vkbInstance, surface);
			vkb::destroy_instance(vkbInstance);
			SDL_DestroyWindow(window);
			SDL_Quit();
			return -1;
		}
		vk::ShaderStageFlagBits stages[] = {
			vk::ShaderStageFlagBits::eVertex,
			vk::ShaderStageFlagBits::eFragment
		};

		vk::ShaderEXT shaders[] = {
			vertShader,
			fragShader
		};


		// ------------------------
		// 8. Main loop
		// ------------------------
		bool running = true;
		SDL_Event event;
		uint32_t currentFrame = 0;

		// Push constant data
		glm::vec2 posOffsets = { 0.0f, 0.0f }; // Start at center

		while (running) {
			while (SDL_PollEvent(&event)) {
				if (event.type == SDL_EVENT_QUIT) running = false;
			}

			vk::SwapchainKHR swapchainHPP(vkbSwapchain.swapchain);

			(void)device.waitForFences(inFlightFences[currentFrame].get(), VK_TRUE, UINT64_MAX);
			(void)device.resetFences(inFlightFences[currentFrame].get());

			uint32_t imageIndex = 0;
			
			vk::Result result = device.acquireNextImageKHR(
				swapchainHPP,             // swapchain
				UINT64_MAX,            // timeout, UINT64_MAX = infinite
				imageAvailableSemaphores[currentFrame].get(), // semaphore to signal
				{},                    // fence (none in this example)
				&imageIndex            // output: index of the acquired image
			);

			if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
				std::cerr << "Failed to acquireNextImageKHR\n";
				return -1;
			}

			// Record command buffer to clear blue
			vk::CommandBuffer cmd = commandBuffers[currentFrame].get();
			(void)cmd.reset();
			(void)cmd.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

			// Transition the swapchain image from UNDEFINED to COLOR_ATTACHMENT_OPTIMAL
			vk::ImageMemoryBarrier2 layoutBarrier;
			layoutBarrier.setSrcStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
				.setSrcAccessMask(vk::AccessFlagBits2::eNone)
				.setDstStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
				.setDstAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
				.setOldLayout(vk::ImageLayout::eUndefined) // We don't care what was there before
				.setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
				.setImage(swapchainImages[imageIndex])
				.setSubresourceRange(vk::ImageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });

			vk::DependencyInfo depInfo;
			depInfo.setImageMemoryBarriers(layoutBarrier);
			cmd.pipelineBarrier2(depInfo);

			// Begin rendering
			vk::ClearValue clearColor(vk::ClearColorValue(std::array<float, 4>{0.0f, 0.2f, 1.0f, 1.0f}));

			vk::RenderingAttachmentInfo colorAttachment{};
			colorAttachment.setImageView(swapchainImageViews[imageIndex])
				.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
				.setLoadOp(vk::AttachmentLoadOp::eClear)
				.setStoreOp(vk::AttachmentStoreOp::eStore)
				.setClearValue(clearColor);

			vk::RenderingInfo renderInfo{};
			renderInfo.setRenderArea({ {0,0},{1280,720} })
				.setLayerCount(1)
				.setColorAttachments(colorAttachment);

			cmd.beginRendering(renderInfo);

			// Bind shaders and draw
			cmd.bindShadersEXT(2, stages, shaders);
			cmd.setPrimitiveTopology(vk::PrimitiveTopology::eTriangleList);

			// Bind vertex buffer
			vk::DeviceSize offset = 0;
			// cmd.bindVertexBuffers(0, 1, &vertexBuffer->getBuffer(), &offset);
			
			cmd.pushConstants<glm::vec2>(
				pipelineLayout,
				vk::ShaderStageFlagBits::eVertex,
				0,
				posOffsets
			);

			const vk::Viewport viewport{ 0, 0, 1280.f, 720.f, 0.f, 1.f };
			const vk::Rect2D rect{ {0,0},{1280,720} };
			cmd.setViewport(0, viewport);
			cmd.setScissor(0, rect);
			cmd.setRasterizerDiscardEnable(false);
			cmd.setCullMode(vk::CullModeFlagBits::eNone);
			cmd.setFrontFace(vk::FrontFace::eCounterClockwise);
			cmd.setStencilTestEnable(false);
			cmd.setDepthTestEnable(false);
			cmd.setDepthWriteEnable(false);
			cmd.setDepthBiasEnable(false);
			cmd.setPolygonModeEXT(vk::PolygonMode::eFill);
			cmd.setRasterizationSamplesEXT(vk::SampleCountFlagBits::e1);
			vk::SampleMask mask = ~0u;  // enable all samples
			cmd.setSampleMaskEXT(vk::SampleCountFlagBits::e1, &mask);
			cmd.setAlphaToCoverageEnableEXT(VK_FALSE);
			cmd.setColorBlendEnableEXT(0, VK_FALSE);
			cmd.setColorWriteMaskEXT(0, vk::ColorComponentFlags(0xF));
			cmd.setViewportWithCount(1, &viewport);
			cmd.setScissorWithCount(1, &rect);
			cmd.setPrimitiveRestartEnable(VK_FALSE);
			vk::VertexInputBindingDescription2EXT binding{};
			binding.binding = 0;
			binding.divisor = 1;
			binding.stride = sizeof(Vertex);
			binding.inputRate = vk::VertexInputRate::eVertex;

			vk::VertexInputAttributeDescription2EXT attribute{};
			attribute.location = 0;
			attribute.binding = 0;
			attribute.format = vk::Format::eR32G32B32Sfloat; // vec3 position
			attribute.offset = offsetof(Vertex, position);

			cmd.setVertexInputEXT(1, &binding, 1, &attribute);

			vk::DeviceSize stride = sizeof(Vertex); // Make sure this >= sum of attribute sizes
			vk::DeviceSize size = sizeof(Vertex) * vertices.size(); // Make sure this >= sum of attribute sizes
			cmd.bindVertexBuffers2(0, 1, &vertexBuffer->getBuffer(), &offset, &size, &stride);
			cmd.draw(3, 1, 0, 0);
			cmd.endRendering();


			// Transition from COLOR_ATTACHMENT_OPTIMAL to PRESENT_SRC_KHR
			layoutBarrier.setSrcStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
				.setSrcAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
				.setDstStageMask(vk::PipelineStageFlagBits2::eBottomOfPipe)
				.setDstAccessMask(vk::AccessFlagBits2::eNone)
				.setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
				.setNewLayout(vk::ImageLayout::ePresentSrcKHR);

			cmd.pipelineBarrier2(depInfo);

			(void)cmd.end();

			// Submit
			vk::CommandBufferSubmitInfo cmdInfo{};
			cmdInfo.setCommandBuffer(cmd);

			vk::SemaphoreSubmitInfo waitInfo{
				imageAvailableSemaphores[currentFrame].get(),
				0,
				vk::PipelineStageFlagBits2::eColorAttachmentOutput
			};

			vk::SemaphoreSubmitInfo signalInfo{
				renderFinishedSemaphores[imageIndex].get(),
				0,
				vk::PipelineStageFlagBits2::eAllCommands
			};

			vk::SubmitInfo2 submit{};
			submit.setCommandBufferInfos(cmdInfo)
				.setWaitSemaphoreInfos(waitInfo)
				.setSignalSemaphoreInfos(signalInfo);

			(void)graphicsQueue.submit2(submit, inFlightFences[currentFrame].get());

			// Present
			vk::PresentInfoKHR present{};
			present.setWaitSemaphores(renderFinishedSemaphores[imageIndex].get())
				.setSwapchains(swapchainHPP)
				.setImageIndices(imageIndex);

			vk::Result presentResult = presentQueue.presentKHR(present);
			if (presentResult != vk::Result::eSuccess && presentResult != vk::Result::eSuboptimalKHR) {
				std::cerr << "Failed to present\n";
			}

			currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

			// Update push constant for animation
			static float time = 0.0f;
			time += 0.01f;
			posOffsets.x = sinf(time) * 0.5f;
			posOffsets.y = cosf(time) * 0.5f;
		}
		(void)device.waitIdle();

		vertexBuffer.reset();

		if (device)
		{
			device.destroyShaderEXT(vertShader);
			device.destroyShaderEXT(fragShader);
			device.destroyPipelineLayout(pipelineLayout);
		}
	}
;
	vmaDestroyAllocator(allocator);
	vkbSwapchain.destroy_image_views(swapchainImageViews);
	vkb::destroy_swapchain(vkbSwapchain);
	vkb::destroy_device(vkbDevice);
	vkb::destroy_surface(vkbInstance, surface);
	vkb::destroy_instance(vkbInstance);
	SDL_DestroyWindow(window);
	SDL_Quit();


	return 0;
}
