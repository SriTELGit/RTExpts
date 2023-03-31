#ifndef MATERIALH
#define MATERIALH

struct hit_record;

#include "ray.h"
#include "hitable.h"

__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
        return false;
}

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}

class material {
public:
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const = 0;
};

class lambertianCuda : public material {
public:
    __device__ lambertianCuda(const vec3& a) : albedo(a) {}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const {
        vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
        scattered = ray(rec.p, target - rec.p);
        attenuation = albedo;
        return true;
    }

    vec3 albedo;
};

class metalCuda : public material {
public:
    __device__ metalCuda(const vec3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }
    vec3 albedo;
    float fuzz;
};



#endif
