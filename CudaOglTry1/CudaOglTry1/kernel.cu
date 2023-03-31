#include <iostream>
#include "Shape.h"
#include "Texture.h"
#include "ShaderClass.h"
#include "Camera.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"



#include <stdio.h>

using namespace std;



const int RT_WIDTH = 1280;
const int RT_HEIGHT = 720;
const float ZNEAR = 0.1f;
const float ZFAR = 100.0f;

class SphInfo {
public:
    SphInfo(glm::vec3 pos, glm::vec4 alb, float r, int t, float rh ) : 
        mSphPos(pos), mAlbedo(alb), mRad(r), mType(t), mRoughness(rh) {}

    glm::vec3 mSphPos;
    glm::vec4 mAlbedo;
    float mRad;
    int mType;
    float mRoughness;
};

vector<SphInfo> gSphInfos;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}



int main()
{
#pragma region OpenGL_code_init_window_creation_vertices_etc
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


    GLfloat vertices[] = {
        -1.0f, -1.0f, 0.0f,  1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        +1.0f, -1.0f, 0.0f,  0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
        +1.0f, +1.0f, 0.0f,  0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
        -1.0f, +1.0f, 0.0f,  1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
    };

    GLuint indices[] = { 0,1,2, 0,2,3};


    GLFWwindow* window = glfwCreateWindow(RT_WIDTH, RT_HEIGHT, "RTWin", NULL, NULL);

    if (window == NULL) { cout << "Window creation failed" << endl; glfwTerminate(); return -1; }

    glfwMakeContextCurrent(window);
    gladLoadGL();

#pragma endregion

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

#pragma region OpenGL_code_viewport_clear_shader_while_poll_clean_up

    

    glm::vec4 botSkyCol = glm::vec4(1, 1, 1, 1);
    glm::vec4 topSkyCol = glm::vec4(0.5, 0.7, 1, 1);

    gSphInfos.push_back(SphInfo(glm::vec3(0, 0, -1),            glm::vec4(0.1, 0.2, 0.5, 1), 0.5, 0, 1.0));
    gSphInfos.push_back(SphInfo(glm::vec3(0, -100.5, -1),       glm::vec4(0.8, 0.8, 0.0, 1), 100, 0, 1.0));
    gSphInfos.push_back(SphInfo(glm::vec3(1, 0, -1),            glm::vec4(0.8, 0.6, 0.2, 1), 0.5, 1, 0.8));


    Shader shdrProgFalseSky("FalseSkyVS.h", "FalseSkyFS.h");
    Shader lambertShdr("lambertVS.h", "lambertFS.h");
    Shader shinyShdr("shinyVS.h", "shinyFS.h");

    Shape triShp(vertices, sizeof(vertices), indices, sizeof(indices));
    Sphere sphShp; sphShp.Create();

    Texture tex0("brick.png");

    float rotation = 0.0f;
    double prevTime = glfwGetTime();
    double currTime = prevTime;

    glm::mat4 model = glm::mat4(1.0f);

    Camera cam(RT_WIDTH, RT_HEIGHT, glm::vec3(0.0f, 0.5f, 2.0f));
    cam.ViewMatrix(); cam.ProjMatrix(45.0f, ZNEAR, ZFAR);
    
    glViewport(0, 0, RT_WIDTH, RT_HEIGHT);

    glClearColor(0.077f, 0.13f, 0.17f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glfwSwapBuffers(window);

    glEnable(GL_DEPTH_TEST);

    while (!glfwWindowShouldClose(window)) {

        currTime = glfwGetTime();

        float dTime = (currTime - prevTime);
        prevTime = currTime;


        glClearColor(0.077f, 0.13f, 0.17f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shdrProgFalseSky.Activate();
        shdrProgFalseSky.SetVec4("botSkyColor", botSkyCol);
        shdrProgFalseSky.SetVec4("topSkyColor", topSkyCol);
        //tex0.ActivateAndBind();
        //shdrProgFalseSky.SetInt("texture0", 0);

        triShp.Draw();

        //draw lambertspheres
        for (int i = 0; i < gSphInfos.size(); i++) {

            model = glm::mat4(1.0f);
            float sc = gSphInfos[i].mRad * 2.0f;
            model = glm::translate(model, gSphInfos[i].mSphPos);
            model = glm::scale(model, glm::vec3(sc,sc,sc));

            if (gSphInfos[i].mType == 0) {

                lambertShdr.Activate();
                lambertShdr.SetMat4("model", model);
                lambertShdr.SetMat4("view", cam.mView);
                lambertShdr.SetMat4("proj", cam.mProj);
                lambertShdr.SetVec4("botSkyColor", botSkyCol);
                lambertShdr.SetVec4("topSkyColor", topSkyCol);
                lambertShdr.SetVec4("albedo", gSphInfos[i].mAlbedo);

            }
            else if (gSphInfos[i].mType == 1) {

                shinyShdr.Activate();
                shinyShdr.SetMat4("model", model);
                shinyShdr.SetMat4("view", cam.mView);
                shinyShdr.SetMat4("proj", cam.mProj);
                shinyShdr.SetVec3("camPosW", cam.mPos);
                shinyShdr.SetVec4("botSkyColor", botSkyCol);
                shinyShdr.SetVec4("topSkyColor", topSkyCol);
                shinyShdr.SetVec4("albedo", gSphInfos[i].mAlbedo);
                shinyShdr.SetFloat("roughness", gSphInfos[i].mRoughness);

            }

            sphShp.Draw();
        }
 


        glfwSwapBuffers(window);


        glfwPollEvents();
        cam.Inputs(window,dTime);
        cam.ViewMatrix();
    }

    triShp.Delete();
    shdrProgFalseSky.Delete();
    tex0.Delete();

    glfwDestroyWindow(window);
    glfwTerminate();

#pragma endregion

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
