#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>

#include "./CudaCode/vec3.h"
#include "./CudaCode/ray.h"
#include "./CudaCode/sphere.h"
#include "./CudaCode/hitable_list.h"
#include "./CudaCode/camera.h"
#include "./CudaCode/material.h"

#include "Shape.h"
#include "Texture.h"
#include "ShaderClass.h"
#include "Camera.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"



#include <stdio.h>

using namespace std;


#pragma region definitions
const int RT_WIDTH = 1280;
const int RT_HEIGHT = 720;
const float ZNEAR = 0.1f;
const float ZFAR = 100.0f;

class SphInfo {
public:
    SphInfo() {}
    void Set(glm::vec3 pos, glm::vec4 alb, float r, int t, float rh) {
        mSphPos = pos; mAlbedo = alb; mRad = r; mType = t; mRoughness = rh;
    }

    glm::vec3 mSphPos;
    glm::vec4 mAlbedo;
    float mRad;
    int mType;
    float mRoughness;
};

const unsigned int NUM_SPH = 9;
SphInfo* gpSphInfos = NULL; // [NUM_SPH] ;

Camera gCam(RT_WIDTH, RT_HEIGHT, glm::vec3(0.0f, 0.5f, 2.0f));
bool gRTDone = false;
Texture* gpRTTex = NULL;

#pragma endregion

#pragma region Cuda_code_global_part1

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, hitable** world, curandState* local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* fb, int max_x, int max_y, int ns, cameraCuda** cam, hitable** world, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

__global__ void create_world(hitable** d_list, hitable** d_world, cameraCuda** d_camera, int nx, int ny, vec3 cp, vec3 cl, vec3 cu, float fov, int sz, SphInfo* sphArr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
       /* d_list[0] = new sphereCuda(vec3(0, 0, -1), 0.5,
            new lambertianCuda(vec3(0.1, 0.2, 0.5)));
        d_list[1] = new sphereCuda(vec3(0, -100.5, -1), 100,
            new lambertianCuda(vec3(0.8, 0.8, 0.0)));
        d_list[2] = new sphereCuda(vec3(1, 0, -1), 0.5,
            new metalCuda(vec3(0.8, 0.6, 0.2), 0.0));*/

        //`SphInfo** sphArr = (SphInfo**)arrIn;

        for (int i = 0; i < sz; i++) {


            SphInfo* spi = (sphArr + i);
            vec3 alb = vec3(spi->mAlbedo.x, spi->mAlbedo.y, spi->mAlbedo.z);

            material* mt = NULL;
            if (spi->mType == 0)
                mt = new lambertianCuda(alb);
            else
                mt = new metalCuda(alb, spi->mRoughness);

            d_list[i] = new sphereCuda(
                vec3(spi->mSphPos.x, spi->mSphPos.y, spi->mSphPos.z),
                spi->mRad, mt);
        }

 
        *d_world = new hitable_list(d_list, sz); 
        *d_camera = new cameraCuda(cp, cl, cu, fov, float(nx) / float(ny));
    }
}

__global__ void free_world(hitable** d_list, hitable** d_world, cameraCuda** d_camera, int sz) {
    for (int i = 0; i < sz; i++) { 
        delete ((sphereCuda*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
}

void CudaRT() {

    int ns = 100;
    int tx = 8;
    int ty = 8;


    std::cerr << "Rendering a " << RT_WIDTH << "x" << RT_HEIGHT << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";
    gRTDone = false;

    int num_pixels = RT_WIDTH * RT_HEIGHT;
    size_t fb_size = num_pixels * sizeof(vec3);

    // allocate FB
    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // allocate random state
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

    // make our world of hitables & the camera
    hitable** d_list; 
    //int numObjInList = NUM_SPH;
    checkCudaErrors(cudaMalloc((void**)&d_list, NUM_SPH * sizeof(hitable*)));



    hitable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
    cameraCuda** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(cameraCuda*)));
    create_world << <1, 1 >> > (d_list, d_world, d_camera, RT_WIDTH, RT_HEIGHT,
        vec3(gCam.mPos.x, gCam.mPos.y, gCam.mPos.z),
        vec3(gCam.mLookAt.x, gCam.mLookAt.y, gCam.mLookAt.z),
        vec3(gCam.mUpAc.x, gCam.mUpAc.y, gCam.mUpAc.z),
        gCam.mFOV, NUM_SPH,  gpSphInfos
        );
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(RT_WIDTH / tx + 1, RT_HEIGHT / ty + 1);
    dim3 threads(tx, ty);
    render_init << <blocks, threads >> > (RT_WIDTH, RT_HEIGHT, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render << <blocks, threads >> > (fb, RT_WIDTH, RT_HEIGHT, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";


    // Output FB as Image
    if (gpRTTex != NULL) 
        delete gpRTTex;
    gpRTTex = new Texture();

    unsigned char* imBytes = (unsigned char*)malloc(num_pixels * 3);

    /*
    ofstream imOut;
    imOut.open("im.ppm");

    if (imOut.is_open()) {

        imOut << "P3\n" << RT_WIDTH << ' ' << RT_HEIGHT << "\n255\n";
        */

        for (int j = RT_HEIGHT-1; j >= 0; j--) {
            for (int i = 0; i < RT_WIDTH; i++) {
                size_t pixel_index = j * RT_WIDTH + i;
                int ir = int(255.99 * fb[pixel_index].r());
                int ig = int(255.99 * fb[pixel_index].g());
                int ib = int(255.99 * fb[pixel_index].b());

                //imOut << ir << ' ' << ig << ' ' << ib << '\n';

                imBytes[3 * pixel_index]     = ir;
                imBytes[3 * pixel_index + 1] = ig;
                imBytes[3 * pixel_index + 2] = ib;
            }
        }

        /*
        imOut.close();

    }
    else {
        cout << "could not open file to write" << endl;
    }
    */

    gpRTTex->FillTexWithData(imBytes, 3, RT_WIDTH, RT_HEIGHT);

    free(imBytes);

    gpRTTex->Unbind();


    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world << <1, 1 >> > (d_list, d_world, d_camera, NUM_SPH);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));

    gRTDone = true;
}


#pragma endregion

#pragma region global_cpp_code

void KeyCallbk(GLFWwindow* pWin, int key, int scancode, int action, int mods)
{

    if (key == GLFW_KEY_R && action == GLFW_PRESS) { gFreezeCamPosForRT = true;  CudaRT(); }
    if (key == GLFW_KEY_U && action == GLFW_PRESS) { gFreezeCamPosForRT = false;  gRTDone = false; }


}

#pragma endregion

int main()
{
#pragma region Cuda_code_part2

/*CUDA prop*/
cudaDeviceProp devProp;
cudaGetDeviceProperties(&devProp, 0);
std::cerr << "Device 0:" << devProp.name << std::endl;

checkCudaErrors( cudaMallocHost( (void**)&gpSphInfos, NUM_SPH*sizeof(SphInfo) ) );
gpSphInfos->Set(        glm::vec3(0, 0, -1),        glm::vec4(0.1, 0.2, 0.5, 1),        0.5, 0, 1.0);
(gpSphInfos + 1)->Set(  glm::vec3(0, -100.5, -1),   glm::vec4(0.8, 0.8, 0.0, 1),        100, 0, 1.0);
(gpSphInfos + 2)->Set(  glm::vec3(1.5, 0, -1),      glm::vec4(0.8, 0.6, 0.2, 1),        0.5, 1, 0.01);
(gpSphInfos + 3)->Set(  glm::vec3(-1.6, 0, -1),     glm::vec4(0.35, 0.47, 0.71, 1),     0.5, 1, 0.25);

(gpSphInfos + 4)->Set(glm::vec3(0.5, -0.25, 0.5),       glm::vec4(0.04, 0.94, 0.55, 1),     0.25, 1, 0.2);
(gpSphInfos + 5)->Set(glm::vec3(-0.5, -0.2, 1.1),      glm::vec4(0.63, 0.27, 0.63, 1),     0.3, 0, 0.3);
(gpSphInfos + 6)->Set(glm::vec3(-1.1, -0.35, 0.6),      glm::vec4(0.98, 0.78, 0.04, 1),     0.15, 1, 0.4);

(gpSphInfos + 7)->Set(glm::vec3(0.3, -0.2, -2.3),      glm::vec4(0.43, 0.59, 0.74, 1),     0.3, 0, 0.1);
(gpSphInfos + 8)->Set(glm::vec3(-0.7, -0.2, -2.4),     glm::vec4(0.78, 0.51, 0.74, 1),     0.25, 1, 0.1);


#pragma endregion

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

    glfwSetKeyCallback(window, KeyCallbk);

    gladLoadGL();

#pragma endregion


#pragma region OpenGL_code_viewport_clear_shader_while_poll_clean_up

    

    glm::vec4 botSkyCol = glm::vec4(1, 1, 1, 1);
    glm::vec4 topSkyCol = glm::vec4(0.5, 0.7, 1, 1);



    Shader shdrProgFalseSky("FalseSkyVS.h", "FalseSkyFS.h");
    Shader lambertShdr("lambertVS.h", "lambertFS.h");
    Shader shinyShdr("shinyVS.h", "shinyFS.h");
    Shader rtTexShdr("rtTexVS.h", "rtTexFS.h");

    Shape triShp(vertices, sizeof(vertices), indices, sizeof(indices));
    Sphere sphShp; sphShp.Create();

    //Texture tex0("brick.png");

    float rotation = 0.0f;
    double prevTime = glfwGetTime();
    double currTime = prevTime;

    glm::mat4 model = glm::mat4(1.0f);

   
    gCam.ViewMatrix(); gCam.ProjMatrix(45.0f, ZNEAR, ZFAR);
    
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
        for (int i = 0; i < NUM_SPH; i++) {

            model = glm::mat4(1.0f);
            float sc = (gpSphInfos + i)->mRad * 2.0f;
            model = glm::translate(model, (gpSphInfos + i)->mSphPos);
            model = glm::scale(model, glm::vec3(sc,sc,sc));

            if ((gpSphInfos + i)->mType == 0) {

                lambertShdr.Activate();
                lambertShdr.SetMat4("model", model);
                lambertShdr.SetMat4("view", gCam.mView);
                lambertShdr.SetMat4("proj", gCam.mProj);
                lambertShdr.SetVec4("botSkyColor", botSkyCol);
                lambertShdr.SetVec4("topSkyColor", topSkyCol);
                lambertShdr.SetVec4("albedo", (gpSphInfos + i)->mAlbedo);

            }
            else if ((gpSphInfos + i)->mType == 1) {

                shinyShdr.Activate();
                shinyShdr.SetMat4("model", model);
                shinyShdr.SetMat4("view", gCam.mView);
                shinyShdr.SetMat4("proj", gCam.mProj);
                shinyShdr.SetVec3("camPosW", gCam.mPos);
                shinyShdr.SetVec4("botSkyColor", botSkyCol);
                shinyShdr.SetVec4("topSkyColor", topSkyCol);
                shinyShdr.SetVec4("albedo", (gpSphInfos + i)->mAlbedo);
                shinyShdr.SetFloat("roughness", (gpSphInfos + i)->mRoughness);

            }

            sphShp.Draw();
        }

        if ((gFreezeCamPosForRT == true) && (gRTDone == true))
        {
            //render quad with RT image texture
            rtTexShdr.Activate();
            gpRTTex->ActivateAndBind();
            rtTexShdr.SetInt("texture0", 0);

            triShp.Draw();
        }
 


        glfwSwapBuffers(window);


        glfwPollEvents();
        gCam.Inputs(window,dTime);
        gCam.ViewMatrix();
    }

    triShp.Delete();
    shdrProgFalseSky.Delete();
    //tex0.Delete();
    if (gpRTTex != NULL)
        delete gpRTTex;

    //delete[] gpSphInfos;

    glfwDestroyWindow(window);
    glfwTerminate();

#pragma endregion

#pragma region cuda_cleanup

checkCudaErrors(cudaFreeHost(gpSphInfos));
cudaDeviceReset();

#pragma endregion

    return 0;
}


