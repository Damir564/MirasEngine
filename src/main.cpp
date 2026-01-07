#include <iostream>
#include <vulkan/vulkan.hpp>
#include <volk.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>
#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>

struct Vertex {
	glm::vec2 position;
};

std::vector<Vertex> vertices = {
	Vertex{{ 0.0f, -0.5f }},
	Vertex{{ 0.5f,  0.5f }},
	Vertex{{-0.5f,  0.5f }},
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
		allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		allocInfo.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
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
		if (m_buffer && m_allocation) {
			vmaDestroyBuffer(m_allocator, VkBuffer(m_buffer), m_allocation);
		}
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
		return -1;
	}

	// ------------------------
	// 2. Initialize volk
	// ------------------------
	if (volkInitialize() != VK_SUCCESS) {
		std::cerr << "volkInitialize failed\n";
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
		.require_api_version(1, 4, 0)
		.use_default_debug_messenger()
		.enable_extensions(extensions)
		// .enable_extension(VK_EXT_SHADER_OBJECT_EXTENSION_NAME)
		.build();

	if (!instRet) {
		std::cerr << "Failed to create instance: " << instRet.error().message() << "\n";
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
		return -1;
	}
	vk::SurfaceKHR surface(surfaceVk);

	// ------------------------
	// 5. Select physical device and create logical device
	// ------------------------
	VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{};
	shaderObjectFeatures.shaderObject = VK_TRUE;

	VkPhysicalDeviceVulkan13Features features13{};
	features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
	features13.synchronization2 = VK_TRUE;
	features13.dynamicRendering = VK_TRUE;
	features13.pNext = &shaderObjectFeatures;

	vkb::PhysicalDeviceSelector selector{ vkbInstance };
	auto physRet = selector
		.set_minimum_version(1, 4)
		.add_required_extension(vk::EXTShaderObjectExtensionName)
		.set_required_features_13(features13)
		.set_surface(surface)
		.select();

	if (!physRet) {
		std::cerr << "Failed to select physical device: " << physRet.error().message() << "\n";
		return -1;
	}


	vkb::PhysicalDevice vkbPhys = physRet.value();
	vk::PhysicalDevice physicalDevice(vkbPhys.physical_device);

	vkb::DeviceBuilder deviceBuilder{ vkbPhys };
	auto deviceRet = deviceBuilder
		// .add_pNext(&shaderObjectFeatures)
		.build();

	if (!deviceRet) {
		std::cerr << "Failed to create device: " << deviceRet.error().message() << "\n";
		return -1;
	}

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

	VmaAllocator allocator;
	if (vmaCreateAllocator(&allocatorInfo, &allocator) != VK_SUCCESS) {
		std::cerr << "Failed to create VMA allocator\n";
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
		return -1;
	}
	vkb::Swapchain vkbSwapchain = swap_ret.value();	
	// ------------------------
	// 11. Get Swapchain Images and ImageViews
	// ------------------------
	auto imagesRet = vkbSwapchain.get_images();
	if (!imagesRet) {
		std::cerr << "Failed to get swapchain images\n";
		return -1;
	}
	std::vector<VkImage> swapchainImages = imagesRet.value();
	uint32_t swapchainImageCount = static_cast<uint32_t>(swapchainImages.size());
	const int MAX_FRAMES_IN_FLIGHT = swapchainImageCount;
	auto viewsRet = vkbSwapchain.get_image_views();
	if (!viewsRet) {
		std::cerr << "Failed to get swapchain image views\n";
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

		// ------ ------- --------- Semaphores and Fences
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

		uint32_t currentFrame = 0;

		// ------------------------
		// Accessing shaders data
		// ------------------------
		auto vertCode = loadSpirv("shaders/triangle.vert.spv");
		auto fragCode = loadSpirv("shaders/triangle.frag.spv");

		std::cout << "Vertex SPIR-V words: " << vertCode.size() << "\n";
		std::cout << "Fragment SPIR-V words: " << fragCode.size() << "\n";

		// shader create info
		vk::ShaderCreateInfoEXT vertInfo{};
		vertInfo.setStage(vk::ShaderStageFlagBits::eVertex)
			.setNextStage(vk::ShaderStageFlagBits::eFragment)
			.setFlags(vk::ShaderCreateFlagBitsEXT::eLinkStage)
			.setCodeType(vk::ShaderCodeTypeEXT::eSpirv)
			.setCodeSize(vertCode.size() * sizeof(vertCode.front()))
			.setPCode(vertCode.data())
			.setPName("main");

		vk::ShaderCreateInfoEXT fragInfo{};
		fragInfo.setStage(vk::ShaderStageFlagBits::eFragment)
			.setFlags(vk::ShaderCreateFlagBitsEXT::eLinkStage)
			.setCodeType(vk::ShaderCodeTypeEXT::eSpirv)
			.setCodeSize(fragCode.size() * sizeof(uint32_t))
			.setPCode(fragCode.data())
			.setPName("main");

		// create shader objects
		auto vertShader = device.createShaderEXT(vertInfo).value;
		auto fragShader = device.createShaderEXT(fragInfo).value;

		// arrays for binding shaders
		vk::ShaderStageFlagBits stages[] = {
			vk::ShaderStageFlagBits::eVertex,
			vk::ShaderStageFlagBits::eFragment
		};

		vk::ShaderEXT shaders[] = {
			vertShader,
			fragShader
		};

		vk::VertexInputBindingDescription bindingDesc{};
		bindingDesc.binding = 0;
		bindingDesc.stride = sizeof(Vertex);
		bindingDesc.inputRate = vk::VertexInputRate::eVertex;

		vk::VertexInputAttributeDescription attrDesc{};
		attrDesc.binding = 0;
		attrDesc.location = 0; // matches shader location
		attrDesc.format = vk::Format::eR32G32Sfloat;
		attrDesc.offset = offsetof(Vertex, position);

		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
		vertexInputInfo.vertexAttributeDescriptionCount = 1;
		vertexInputInfo.pVertexAttributeDescriptions = &attrDesc;

		// Add pushConstantsRange

		vk::PushConstantRange pcRange{};
		pcRange.stageFlags = vk::ShaderStageFlagBits::eVertex;
		pcRange.offset = 0;
		pcRange.size = sizeof(glm::vec2); // vec2

		vk::PipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.setPushConstantRanges(pcRange);

		vk::PipelineLayout pipelineLayout =
			device.createPipelineLayout(layoutInfo).value;

		vk::PipelineDynamicStateCreateInfo pipelineDynamicStateCreateInfo{};

		vk::GraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.setLayout(pipelineLayout);
		pipelineInfo.setPVertexInputState(&vertexInputInfo);
		pipelineInfo.setPDynamicState(&pipelineDynamicStateCreateInfo);

		// variables
		glm::vec2 posOffsets = { -0.5f, -0.2f }; // move triangle
		VertexBuffer vertexBuffer(allocator, device, vertices);

		// ------------------------
		// 8. Main loop (empty)
		// ------------------------
		bool running = true;
		SDL_Event event;
		//uint32_t currentFrame = 0;
		while (running) {
			while (SDL_PollEvent(&event)) {
				if (event.type == SDL_EVENT_QUIT) running = false;
			}

			vk::SwapchainKHR swapchainHPP(vkbSwapchain.swapchain);

			uint32_t imageIndex = 0;

			(void)device.waitForFences(inFlightFences[currentFrame].get(), VK_TRUE, UINT64_MAX);
			(void)device.resetFences(inFlightFences[currentFrame].get());
			
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
			vk::DeviceSize offset = 0;
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

			cmd.bindShadersEXT(2, stages, shaders);
			cmd.setPrimitiveTopology(vk::PrimitiveTopology::eTriangleList);
			cmd.bindVertexBuffers(0, 1, &vertexBuffer.getBuffer(), &offset);
			cmd.pushConstants<glm::vec2>(
				pipelineLayout,
				vk::ShaderStageFlagBits::eVertex,
				0,
				posOffsets
			);
			cmd.setViewport(0, vk::Viewport{ 0, 0, 1280.f, 720.f, 0.f, 1.f });
			cmd.setScissor(0, vk::Rect2D{ {0,0},{1280,720} });


			// Transition from COLOR_ATTACHMENT_OPTIMAL to PRESENT_SRC_KHR
			layoutBarrier.setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
				.setNewLayout(vk::ImageLayout::ePresentSrcKHR)
				.setSrcAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
				.setDstAccessMask(vk::AccessFlagBits2::eNone);

			cmd.pipelineBarrier2(depInfo);


			cmd.draw(vertexBuffer.getVertexCount(), 1, 0, 0);

			cmd.endRendering();

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

			(void)presentQueue.presentKHR(present);
			currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
		}
		if (device)
		{
			(void)device.waitIdle();
			device.destroyShaderEXT(vertShader);
			device.destroyShaderEXT(fragShader);
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
