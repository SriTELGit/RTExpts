#ifndef SHAPE_H
#define SHAPE_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


#include <vector>
using namespace std;

const float PI = 3.14159265359f;
const float TWO_PI = 6.28318530718f;


const unsigned int X_SEG2 = 20; 
const unsigned int Y_SEG2 = 20; 

class VBO {
public:
	GLuint mId;
	VBO(GLfloat* verts, GLsizeiptr size) {
		glGenBuffers(1, &mId);
		glBindBuffer(GL_ARRAY_BUFFER, mId);
		glBufferData(GL_ARRAY_BUFFER, size, verts, GL_STATIC_DRAW);
	}

	void Bind() { glBindBuffer(GL_ARRAY_BUFFER, mId); }
	void Unbind() { glBindBuffer(GL_ARRAY_BUFFER, 0); }
	void Delete() { glDeleteBuffers(1, &mId); }
};

class EBO {
public:
	GLuint mId;
	EBO(GLuint* indices, GLsizeiptr indSize ) {
		glGenBuffers(1, &mId);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mId);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indSize, indices, GL_STATIC_DRAW);
	}

	void Bind() { glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mId); }
	void Unbind() { glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); }
	void Delete() { glDeleteBuffers(1, &mId); }

};

class VAO {
public:
	GLuint mId;

	VAO() {
		glGenVertexArrays(1, &mId);
		glBindVertexArray(mId);
	}

	void Bind(){ glBindVertexArray(mId); }
	void Unbind(){ glBindVertexArray(0); }
	void Delete(){ glDeleteVertexArrays(1, &mId); }
};

class Shape {
public:
	VAO mVAO; VBO mVBO; EBO mEBO;
	int mNumIndices;

	Shape(GLfloat* verts, GLsizeiptr size, GLuint* indices, GLsizeiptr indSize) : 
		mVAO(), mVBO(verts, size), mEBO(indices, indSize)
	{
		mNumIndices = indSize;

		//glVertexAttribPointer(layoutPos, numComponents, type, normalize, stride, offset);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);

		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)12);
		glEnableVertexAttribArray(1);

		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)24);
		glEnableVertexAttribArray(2);


		mVBO.Unbind(); mVAO.Unbind(); mEBO.Unbind();

	}

	
	void Draw() {
		mVAO.Bind();
		//glDrawArrays(GL_TRIANGLES, 0, 3);
		glDrawElements(GL_TRIANGLES, mNumIndices, GL_UNSIGNED_INT, 0);

	}

	void Delete() {
		mVAO.Delete(); mVBO.Delete(); mEBO.Delete();
	}
	
};

class Shape2 {

protected:
	unsigned int mVAO;
	unsigned int mIndexCount;
	unsigned int mVBO, mEBO;
	unsigned int mStride;
	vector<float> mData;
	vector<unsigned int> mIndices;

public:

	Shape2() {
		mVAO = 0;
		mIndexCount = 0;
		mVBO = 0;
		mEBO = 0;
		mStride = 0;
		mData.clear();
		mIndices.clear();
	}

	virtual ~Shape2() {

		mData.clear();
		mIndices.clear();

		glDeleteBuffers(1, &mVBO);
		glDeleteBuffers(1, &mEBO);
		glDeleteVertexArrays(1, &mVAO);
	}

	virtual void FillData() {}

	virtual void Draw() {}

	virtual void Create() {
		glGenVertexArrays(1, &mVAO);

		glGenBuffers(1, &mVBO); glGenBuffers(1, &mEBO);

		FillData();

		glBindVertexArray(mVAO);
		glBindBuffer(GL_ARRAY_BUFFER, mVBO);
		glBufferData(GL_ARRAY_BUFFER, mData.size() * sizeof(float), &mData[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mEBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, mIndexCount * sizeof(unsigned int), &mIndices[0], GL_STATIC_DRAW);

		//glVertexAttribPointer(layoutPos, numComponents, type, normalize, stride, offset);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, mStride * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, mStride * sizeof(float), (void*)(3 * sizeof(float)));


	}


};

class Sphere : public Shape2 {

public:
	 
	Sphere() : Shape2() {}

	virtual void FillData() override {

		std::vector<glm::vec3> posTemp;
		std::vector<glm::vec3> nTemp;

		glm::vec3 pos;
		glm::vec3 nt;

		for (unsigned int y = 0; y <= Y_SEG2; y++) {

			if (y == 0) {
				pos = glm::vec3(0, +0.5, 0);
				nt = glm::vec3(0, 1, 0);

				posTemp.push_back(pos);
				nTemp.push_back(nt);
			}
			else if (y == Y_SEG2) {
				pos = glm::vec3(0, -0.5, 0);
				nt = glm::vec3(0, -1, 0);

				posTemp.push_back(pos);
				nTemp.push_back(nt);
			}
			else {

				for (unsigned int x = 0; x <= X_SEG2; x++) {

					float xt = float(x) / X_SEG2;
					float yt = float(y) / Y_SEG2;
					pos = glm::vec3(sin(yt * PI) * cos(xt * TWO_PI), cos(yt * PI), sin(yt * PI) * sin(xt * TWO_PI)) * 0.5f;
					nt = glm::normalize(pos);

					posTemp.push_back(pos);
					nTemp.push_back(nt);

				}

			}

		}

		int numVerts = posTemp.size();
		unsigned int currIndx = 2;

		for (unsigned int y = 0; y < Y_SEG2; y++) {

			if (y == 0) {

				for (unsigned int x = 1; x <= X_SEG2; x++) {
					mIndices.push_back(0);
					mIndices.push_back(currIndx);
					mIndices.push_back(currIndx - 1);

					currIndx++;
				}
			}
			else if (y == (Y_SEG2 - 1)) {

				currIndx = numVerts - 1 - X_SEG2 - 1;

				for (unsigned int x = 0; x < X_SEG2; x++) {
					mIndices.push_back(currIndx);
					mIndices.push_back(currIndx + 1);
					mIndices.push_back(numVerts - 1);

					currIndx++;
				}
			}
			else {
				for (unsigned int x = 0; x < X_SEG2; x++) {
					unsigned int com = currIndx - X_SEG2 - 1;
					mIndices.push_back(com);
					mIndices.push_back(currIndx + 1);
					mIndices.push_back(currIndx);

					mIndices.push_back(com);
					mIndices.push_back(com + 1);
					mIndices.push_back(currIndx + 1);

					currIndx++;

				}
				currIndx++;
			}
		}

		int numTris = mIndices.size() / 3;

		mIndexCount = mIndices.size();

		mStride = (3 + 3);

		for (unsigned int i = 0; i < posTemp.size(); i++) {
			mData.push_back(posTemp[i].x); mData.push_back(posTemp[i].y); mData.push_back(posTemp[i].z);

			mData.push_back(nTemp[i].x); mData.push_back(nTemp[i].y); mData.push_back(nTemp[i].z);
		}



	}

	virtual void Draw() override {

		glBindVertexArray(mVAO);
		glDrawElements(GL_TRIANGLES, mIndexCount, GL_UNSIGNED_INT, 0);

	}

};


#endif
