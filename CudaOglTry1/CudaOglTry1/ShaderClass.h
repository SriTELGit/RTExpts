#ifndef SHDR_CLASS_H
#define SHDR_CLASS_H

#include <glad/glad.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cerrno>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


std::string GetFileContents(const char* fileName) {

	std::ifstream in(fileName, std::ios::binary);

	if (in) {
		std::string contents;
		in.seekg(0, std::ios::end);
		contents.resize(in.tellg());
		in.seekg(0, std::ios::beg);
		in.read(&contents[0], contents.size());
		in.close();
		return(contents);
	}
}

class Shader {
public:

	GLuint mId;
	Shader(const char* vFile, const char* fFile) {

		std::string vsCode = GetFileContents(vFile);
		std::string fsCode = GetFileContents(fFile);

		const char* vsSrc = vsCode.c_str();
		const char* fsSrc = fsCode.c_str();

		GLuint vShdr = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vShdr, 1, &vsSrc, NULL);
		glCompileShader(vShdr);
		checkCompileErrors(vShdr, "VERTEX");
		GLuint fShdr = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fShdr, 1, &fsSrc, NULL);
		glCompileShader(fShdr);
		checkCompileErrors(fShdr, "FRAGMENT");
		mId = glCreateProgram();
		glAttachShader(mId, vShdr); glAttachShader(mId, fShdr);
		glLinkProgram(mId);
		checkCompileErrors(mId, "PROGRAM");
		glDeleteShader(vShdr); glDeleteShader(fShdr);

	}

	void Activate() {
		glUseProgram(mId);
	}
	void Delete() {
		glDeleteProgram(mId);
	}

	void SetFloat(const std::string& name, float value) const
	{
		glUniform1f(glGetUniformLocation(mId, name.c_str()), value);
	}

	void SetInt(const std::string& name, int value) const
	{
		glUniform1i(glGetUniformLocation(mId, name.c_str()), value);
	}

	void SetMat4(const std::string& name, const glm::mat4& mat) const
	{
		glUniformMatrix4fv(glGetUniformLocation(mId, name.c_str()), 1, GL_FALSE, &mat[0][0]);
	}

	void SetVec4(const std::string& name, const glm::vec4& value) const
	{
		glUniform4fv(glGetUniformLocation(mId, name.c_str()), 1, &value[0]);
	}

	void SetVec3(const std::string& name, const glm::vec3& value) const
	{
		glUniform3fv(glGetUniformLocation(mId, name.c_str()), 1, &value[0]);
	}


	void checkCompileErrors(GLuint shader, std::string type)
	{
		GLint success;
		GLchar infoLog[1024];
		if (type != "PROGRAM")
		{
			glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
			if (!success)
			{
				glGetShaderInfoLog(shader, 1024, NULL, infoLog);
				std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
			}
		}
		else
		{
			glGetProgramiv(shader, GL_LINK_STATUS, &success);
			if (!success)
			{
				glGetProgramInfoLog(shader, 1024, NULL, infoLog);
				std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
			}
		}
	}

};

#endif