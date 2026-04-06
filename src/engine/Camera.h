#pragma once

struct Camera {
	glm::vec3 position{ 0.0f, 0.0f, 0.0f };
	float yaw = -90.0f;
	float pitch = 0.0f;
	float speed = 10.0f;
	float sensitivity = 0.1f;
};

glm::mat4 getView(const Camera& cam) {
	glm::vec3 front{
		cos(glm::radians(cam.yaw)) * cos(glm::radians(cam.pitch)),
		sin(glm::radians(cam.pitch)),
		sin(glm::radians(cam.yaw)) * cos(glm::radians(cam.pitch))
	};

	return glm::lookAt(
		cam.position,
		cam.position + glm::normalize(front),
		glm::vec3(0, 1, 0)
	);
}

glm::mat4 getProjection(float width, float height) {
	glm::mat4 proj = glm::perspective(
		glm::radians(60.0f),
		width / height,
		0.1f,
		10000.0f
	);
	proj[1][1] *= -1; // Vulkan clip space fix
	return proj;
}

static bool isAABBVisible(const glm::vec3& minB, const glm::vec3& maxB, const glm::mat4& MVP) {
	glm::vec4 corners[8] = {
		{minB.x, minB.y, minB.z, 1.0f}, {maxB.x, minB.y, minB.z, 1.0f},
		{minB.x, maxB.y, minB.z, 1.0f}, {maxB.x, maxB.y, minB.z, 1.0f},
		{minB.x, minB.y, maxB.z, 1.0f}, {maxB.x, minB.y, maxB.z, 1.0f},
		{minB.x, maxB.y, maxB.z, 1.0f}, {maxB.x, maxB.y, maxB.z, 1.0f}
	};

	int outXMin = 0, outXMax = 0, outYMin = 0, outYMax = 0, outZMin = 0, outZMax = 0;

	for (int i = 0; i < 8; i++) {
		glm::vec4 pt = MVP * corners[i];
		if (pt.x < -pt.w) outXMin++;
		if (pt.x > pt.w) outXMax++;
		if (pt.y < -pt.w) outYMin++;
		if (pt.y > pt.w) outYMax++;
		if (pt.z < 0.0f) outZMin++; // Vulkan uses 0 to w for Depth
		if (pt.z > pt.w) outZMax++;
	}

	// If ALL 8 corners are outside any single plane of the camera view, it's invisible
	if (outXMin == 8 || outXMax == 8 || outYMin == 8 || outYMax == 8 || outZMin == 8 || outZMax == 8) {
		return false;
	}
	return true; // At least partially visible
}