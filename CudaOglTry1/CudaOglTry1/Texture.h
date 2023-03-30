#include <stb/stb_image.h>

class Texture {
public:
	int mWidth, mHeight, mNumColCh;
	unsigned int mTexId;

	Texture(const char* texFile) {
		stbi_set_flip_vertically_on_load(true);
		unsigned char* bytes = stbi_load(texFile, &mWidth, &mHeight, &mNumColCh, 0);

		glGenTextures(1, &mTexId);
		ActivateAndBind();

		glSamplerParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glSamplerParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, mWidth, mHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, bytes);
		glGenerateMipmap(GL_TEXTURE_2D);

		stbi_image_free(bytes);
		Unbind();
		
	}

	void ActivateAndBind() {
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, mTexId);
	}

	void Unbind() { glBindTexture(GL_TEXTURE_2D, 0); }

	void Delete() { glDeleteTextures(1, &mTexId); }
};
