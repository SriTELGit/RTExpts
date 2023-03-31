#include <stb/stb_image.h>

class Texture {
public:
	int mWidth, mHeight, mNumColCh;
	unsigned int mTexId;

	Texture() {

		glGenTextures(1, &mTexId);
		ActivateAndBind();

		SetSampleParams();

	}

	~Texture() { Delete(); }

	Texture(const char* texFile) {
		stbi_set_flip_vertically_on_load(true);
		unsigned char* bytes = stbi_load(texFile, &mWidth, &mHeight, &mNumColCh, 0);

		glGenTextures(1, &mTexId);
		ActivateAndBind();

		SetSampleParams();

		FillTexWithData(bytes, mNumColCh, mWidth, mHeight);

		stbi_image_free(bytes);
		Unbind();
		
	}

	void SetSampleParams() {

		glSamplerParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glSamplerParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	}

	void FillTexWithData(unsigned char* bytes, int nColCh, int w, int h) {
		if (nColCh == 4) {
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, bytes);
		}
		else {
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, bytes);
		}
		glGenerateMipmap(GL_TEXTURE_2D);
	}

	void ActivateAndBind() {
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, mTexId);
	}

	void Unbind() { glBindTexture(GL_TEXTURE_2D, 0); }

	void Delete() { 
		glDeleteTextures(1, &mTexId); 
	}
};
