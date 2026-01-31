#include "VulkanResources.h"

VulkanBuffer::VulkanBuffer(VulkanContext& ctx, vk::DeviceSize size, vk::BufferUsageFlags usage, VmaMemoryUsage memoryUsage)
    : ctx(&ctx)
{
    VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = size;
    bufferInfo.usage = (VkBufferUsageFlags)usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = memoryUsage;
    if (memoryUsage == VMA_MEMORY_USAGE_CPU_TO_GPU)
        allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    VkBuffer rawBuffer;
    if (vmaCreateBuffer(ctx.getAllocator(), &bufferInfo, &allocInfo, &rawBuffer, &allocation, nullptr) != VK_SUCCESS)
        throw std::runtime_error("Failed to create buffer");
    buffer = vk::Buffer(rawBuffer);
}

VulkanBuffer::~VulkanBuffer() {
    if (buffer) vmaDestroyBuffer(ctx->getAllocator(), VkBuffer(buffer), allocation);
}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept
    : ctx(other.ctx), buffer(other.buffer), allocation(other.allocation), count(other.count) {
    other.buffer = nullptr; other.allocation = nullptr;
}

void VulkanBuffer::upload(const void* data, size_t size) {
    void* mapped;
    vmaMapMemory(ctx->getAllocator(), allocation, &mapped);
    memcpy(mapped, data, size);
    vmaUnmapMemory(ctx->getAllocator(), allocation);
}

// --------------------------------------------------------------------------------------

VulkanTexture::VulkanTexture(VulkanContext& ctx, const TextureData& data, vk::Format format)
    : ctx(&ctx)
{
    vk::DeviceSize imageSize = data.width * data.height * 4;

    // 1. Staging Buffer
    VulkanBuffer staging(ctx, imageSize, vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_CPU_ONLY);
    staging.upload(data.pixels, imageSize);

    // 2. Create GPU Image
    VkImageCreateInfo imageInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = { (uint32_t)data.width, (uint32_t)data.height, 1 };
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = (VkFormat)format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    VmaAllocationCreateInfo imgAllocInfo = {};
    imgAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkImage rawImg;
    if (vmaCreateImage(ctx.getAllocator(), &imageInfo, &imgAllocInfo, &rawImg, &allocation, nullptr) != VK_SUCCESS)
        throw std::runtime_error("Failed to create image");
    image = vk::Image(rawImg);

    // 3. Upload & Transition
    ctx.immediateSubmit([&](vk::CommandBuffer cmd) {
        vk::ImageMemoryBarrier barrier{};
        barrier.oldLayout = vk::ImageLayout::eUndefined;
        barrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eNone;
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.image = image;
        barrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };

        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 0, nullptr, 1, &barrier);

        vk::BufferImageCopy copyRegion{};
        copyRegion.imageSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 };
        copyRegion.imageExtent = vk::Extent3D{ (uint32_t)data.width, (uint32_t)data.height, 1 };
        cmd.copyBufferToImage(staging.getBuffer(), image, vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);

        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, 0, nullptr, 0, nullptr, 1, &barrier);
        });

    // 4. Create View
    vk::ImageViewCreateInfo viewInfo{};
    viewInfo.image = image;
    viewInfo.viewType = vk::ImageViewType::e2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
    view = ctx.getDevice().createImageView(viewInfo).value;
}

VulkanTexture::VulkanTexture(VulkanTexture&& other) noexcept
    : ctx(other.ctx), image(other.image), allocation(other.allocation), view(other.view) {
    other.image = nullptr; other.view = nullptr; other.allocation = nullptr;
}

VulkanTexture::~VulkanTexture() {
    if (view) ctx->getDevice().destroyImageView(view);
    if (image) vmaDestroyImage(ctx->getAllocator(), VkImage(image), allocation);
}