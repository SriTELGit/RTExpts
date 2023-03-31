#ifndef CAMERA_H
#define CAMERA_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/vector_angle.hpp>

bool gFreezeCamPosForRT = false;

class Camera {
public:
	glm::vec3 mPos;
	glm::vec3 mLookAt = glm::vec3(0, 0, -1);
	glm::vec3 mUp = glm::vec3(0, 1, 0);
	glm::vec3 mUpAc = glm::vec3(0, 1, 0);

	float mSpeed = 5.0f;
	float mSensitivity = 5000.0f;
	float mFOV = 45.0f;

	int mWidth, mHeight;

	glm::mat4 mView;
	glm::mat4 mProj;

	bool mFirstClick = true;

	Camera(int w, int h, glm::vec3 pos) {
		mWidth = w;
		mHeight = h;
		mPos = pos;
	}

	

	void ProjMatrix(float fovDeg, float nearPlane, float farPlane) {

		mFOV = fovDeg;

		mProj = glm::perspective(glm::radians(fovDeg), (float)mWidth / (float)mHeight, nearPlane, farPlane);

	}

	void ViewMatrix() {

		mView = glm::lookAt(mPos, mPos + mLookAt, mUpAc);
	}

	void Inputs(GLFWwindow* win, float dt) {


		if (gFreezeCamPosForRT == true) return;

		glm::vec3 rVec = glm::normalize(glm::cross(mLookAt, mUp));

		float ds = mSpeed * dt;
		float dr = mSensitivity * dt;

		if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) {
			mPos += ds * mLookAt;
		}
		if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) {
			mPos -= ds * mLookAt;
		}

		if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) {
			mPos -= ds * rVec;
		}
		if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) {
			mPos += ds * rVec;
		}

		if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {

			glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

			if (mFirstClick) {
				glfwSetCursorPos(win, mWidth / 2, mHeight / 2);
				mFirstClick = false;
			}

			double mousex, mousey;
			glfwGetCursorPos(win, &mousex, &mousey);

			float rotX = dr * (float)(mousey - mHeight / 2) / (float)mHeight;
			float rotY = dr * (float)(mousex - mWidth / 2) / (float)mWidth;

			glm::vec3 newLookAt = glm::rotate(mLookAt, glm::radians(-rotX), rVec);

			bool bNearUp = (glm::angle(newLookAt, mUp) <= glm::radians(5.0f)) || (glm::angle(newLookAt, -mUp) <= glm::radians(5.0f));

			if (!bNearUp)
				mLookAt = newLookAt;

			mLookAt = glm::rotate(mLookAt, glm::radians(-rotY), mUp);

			rVec = glm::normalize(glm::cross(mLookAt, mUp));
			mUpAc = glm::normalize(glm::cross(rVec, mLookAt));

			glfwSetCursorPos(win, mWidth / 2, mHeight / 2);

		}

		if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) {

			glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			mFirstClick = true;

		}


	}
};

#endif
