#ifndef SHAPE_H
#define SHAPE_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

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

#endif
