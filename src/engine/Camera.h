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

struct FrustumPlanes {
	glm::vec4 planes[6];
};

// Gribb-Hartmann extraction; planes are in the space the matrix maps from (e.g. model space for P*V*M).
static FrustumPlanes extractFrustumPlanes(const glm::mat4& m) {
	glm::vec4 r0(m[0][0], m[1][0], m[2][0], m[3][0]);
	glm::vec4 r1(m[0][1], m[1][1], m[2][1], m[3][1]);
	glm::vec4 r2(m[0][2], m[1][2], m[2][2], m[3][2]);
	glm::vec4 r3(m[0][3], m[1][3], m[2][3], m[3][3]);
	return { { r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2 } };
}

static bool isAABBInFrustum(const FrustumPlanes& f, const glm::vec3& minB, const glm::vec3& maxB) {
	for (const glm::vec4& p : f.planes) {
		glm::vec3 positive(p.x >= 0.0f ? maxB.x : minB.x,
			p.y >= 0.0f ? maxB.y : minB.y,
			p.z >= 0.0f ? maxB.z : minB.z);
		if (p.x * positive.x + p.y * positive.y + p.z * positive.z + p.w < 0.0f)
			return false;
	}
	return true;
}