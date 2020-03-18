#define TINYOBJLOADER_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define RAYS_PER_PIXEL 4

#include "tiny_obj_loader.h"

#include <boost/progress.hpp>
#include <limits>
#include <cmath>
#include <iostream>
#include <vector>
#include "geometry.h"
#include "stb_image.h"

int32_t cubemap_width, cubemap_height, cubemap_channels;
uint8_t *cubemap_neg_x_raw, *cubemap_neg_y_raw, *cubemap_neg_z_raw, *cubemap_pos_x_raw, *cubemap_pos_y_raw, *cubemap_pos_z_raw;

struct Light {
    Light(const Vec3f &p, const float &i) : position(p), intensity(i) {}
    Vec3f position;
    float intensity;
};

struct Material {
    Material(const float &r, const Vec4f &a,
            const Vec3f &color, const float &spec) : refractive_index(r), \
    albedo(a), diffuse_color(color), specular_exponent(spec) {}
    Material() : refractive_index(1), albedo(1,0,0,0), \
            diffuse_color(), specular_exponent() {}
    float refractive_index;
    Vec4f albedo;
    Vec3f diffuse_color;
    float specular_exponent;
};

struct Ray
{
    Vec3f origin;
    Vec3f direction;
    Vec3f inverse_direction;
    Vec3i sign;
    Ray(const Vec3f &o, const Vec3f &d): origin(o) {
        direction = d;
        direction = direction.normalize();
        inverse_direction = {1.f / direction.x,
                             1.f / direction.y, direction.z };
        sign = {inverse_direction.x < 0,
                inverse_direction.y < 0,inverse_direction.z < 0};
    }
};

struct Sphere {
    Vec3f center;
    float radius;
    Material material;

    Sphere(const Vec3f &c, const float &r,
            const Material &m) : center(c), radius(r), material(m) {}

    // D = direction, norm(D) = 1
    // orig - camera point
    // t_1, t_2 = <L, D> +- sqrt(<L, D>^2 - <L, L> + r^2)
    // returns true if ray intersect sphere (+ intersect point), else returns false
    bool ray_intersect(const Ray &r, float &intersect_point) const {
        auto orig = r.origin;
        auto direction = r.direction;
        Vec3f L = center - orig;
        float L_D = L * direction;
        float discriminant = L_D * L_D - L * L + radius * radius;
        if (discriminant < 0) return false;
        float disc_sqrt = sqrtf(discriminant);
        float t1 = L_D - disc_sqrt;
        float t2 = L_D + disc_sqrt;
        if (t1 < 0) {
            intersect_point = t2;
            return false;
        } else {
            intersect_point = t1;
            return true;
        }
    }
};

struct Triangle
{
    Vec3f x, y, z;
    Material material;
    Triangle (const Vec3f &x, const Vec3f &y, const Vec3f &z,
            const Material &m): x(x), y(y), z(z), material(m) {}
};

bool triangle_intersection(const Ray &r, const Vec3f &x, const Vec3f &y,
        const Vec3f &z, float &distance) {
    Vec3f e1 = y - x;
    Vec3f e2 = z - x;
    Vec3f pvec = cross(r.direction, e2);
    float det = (e1 * pvec);
    if (det < 1e-3f) {
        return false;
    }
    float inv_det = 1.00f / det;
    Vec3f tvec = r.origin - x;
    float u = (tvec * pvec) * inv_det;
    if (u < 0.00f || u > 1.00f) {
        return false;
    }
    Vec3f qvec = cross(tvec, e1);
    float v = (r.direction * qvec) * inv_det;
    if (v < 0.00f || u + v > 1.00f) {
        return false;
    }
    distance = (e2 * qvec) * inv_det;
    return distance > 1e-3f;
}

struct Model
{
    Vec3f bounds[2];
    Material material;
    std::string err, warn;
    std::vector<tinyobj::material_t> materials;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<Triangle> triangles;
    tinyobj::attrib_t attrib;
    Model() = default;
    Model(const char *s, const Material &m, const Vec3f &shift): material(m) {
        bool ret = tinyobj::LoadObj(&attrib, &shapes,
                &materials, &warn, &err, s, nullptr,
                true,true);
        if (!ret) {
            std::cout << warn << std::endl;
            std::cout << err << std::endl;
            return;
        }
        bounds[0] = Vec3f(1000.f, 1000.f, 1000.f);
        bounds[1] = Vec3f(-1000.f, -1000.f, -1000.f);
        for (uint32_t i = 0; i < attrib.vertices.size(); i += 3) {
            attrib.vertices[i] += shift.x;
            attrib.vertices[i + 1] += shift.y;
            attrib.vertices[i + 2] += shift.z;
            float x = attrib.vertices[i];
            float y = attrib.vertices[i + 1];
            float z = attrib.vertices[i + 2];
            if (x < bounds[0].x) {
                bounds[0].x = x;
            }
            if (x > bounds[1].x) {
                bounds[1].x = x;
            }
            if (y < bounds[0].y) {
                bounds[0].y = y;
            }
            if (y > bounds[1].y) {
                bounds[1].y = y;
            }
            if (z < bounds[0].z) {
                bounds[0].z = z;
            }
            if (z > bounds[1].z) {
                bounds[1].z = z;
            }
        }
    }
    bool intersection(const Ray &r) const {
        float tmin, tmax, tymin, tymax, tzmin, tzmax;

        tmin = (bounds[r.sign[0]].x - r.origin.x) * r.inverse_direction.x;
        tmax = (bounds[1 - r.sign[0]].x - r.origin.x) * r.inverse_direction.x;
        tymin = (bounds[r.sign[1]].y - r.origin.y) * r.inverse_direction.y;
        tymax = (bounds[1 - r.sign[1]].y - r.origin.y) * r.inverse_direction.y;

        if ((tmin > tymax) || (tymin > tmax)) {
            return false;
        }
        if (tymin > tmin) {
            tmin = tymin;
        }
        if (tymax < tmax) {
            tmax = tymax;
        }

        tzmin = (bounds[r.sign[2]].z - r.origin.z) * r.inverse_direction.z;
        tzmax = (bounds[1 - r.sign[2]].z - r.origin.z) * r.inverse_direction.z;

        return !((tmin > tzmax) || (tzmin > tmax));
    }
};

std::vector<struct Model> models;
std::vector<Sphere> spheres;
std::vector<Light> lights;
std::vector<Triangle> triangles;

// R = 2 * N * <N, L> - L
Vec3f reflect(const Vec3f &I, const Vec3f &N) {
    return  I - N * 2.f * (I * N);
}

Vec3f refract(const Vec3f &I, const Vec3f &N,
        const float eta_t, const float eta_i = 1.f) { // Snell's law
    float cosi = - std::max(-1.f, std::min(1.f, I * N));
    if (cosi < 0) {
        // if the ray comes from the inside the object, swap the air and the media
        return refract(I, -N, eta_i, eta_t);
    }
    float eta = eta_i / eta_t;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    // k<0 = total reflection, no ray to refract. I refract it anyways, this has no physical meaning
    return k < 0 ? Vec3f(1,0,0) : I * eta + N * (eta * cosi - sqrtf(k));
}

// D = direction, norm(D) = 1
// orig - camera point
// N - normal vector
// hit - intersection point
// N = hit - center / |hit - center|
bool scene_intersect(const Ray &r, Vec3f &hit, Vec3f &N, Material &material) {
    auto orig = r.origin;
    auto dir = r.direction;
    float spheres_dist = std::numeric_limits<float>::max();
    for (const auto & sphere : spheres) {
            float dist_i;
            if (sphere.ray_intersect(r, dist_i) && dist_i < spheres_dist) {
                spheres_dist = dist_i;
                hit = orig + dir * dist_i;
                N = (hit - sphere.center).normalize();
                material = sphere.material;
            }
    }

    float model_distance = 100000.f;
    for (const auto &model: models) {
        if (model.intersection(r)) {
            for (uint32_t f = 0, offset = 0; f < model.shapes[0].mesh.num_face_vertices.size(); ++f, offset += 3) {
                float d;
                Vec3f x(model.attrib.vertices[3 * model.shapes[0].mesh.indices[offset].vertex_index],
                            model.attrib.vertices[3 * model.shapes[0].mesh.indices[offset].vertex_index + 1],
                            model.attrib.vertices[3 * model.shapes[0].mesh.indices[offset].vertex_index + 2]);
                Vec3f y(model.attrib.vertices[3 * model.shapes[0].mesh.indices[offset + 1].vertex_index],
                            model.attrib.vertices[3 * model.shapes[0].mesh.indices[offset + 1].vertex_index + 1],
                            model.attrib.vertices[3 * model.shapes[0].mesh.indices[offset + 1].vertex_index + 2]);
                Vec3f z(model.attrib.vertices[3 * model.shapes[0].mesh.indices[offset + 2].vertex_index],
                            model.attrib.vertices[3 * model.shapes[0].mesh.indices[offset + 2].vertex_index + 1],
                            model.attrib.vertices[3 * model.shapes[0].mesh.indices[offset + 2].vertex_index + 2]);
                if (triangle_intersection(r, x, y, z, d) && d < model_distance) {
                    model_distance = d;
                    hit = orig + dir * model_distance;
                    N = cross(x - y, y - z).normalize();
                    material = model.material;
                }
                model.shapes[0].mesh.material_ids[f];
            }
        }
    }

    float checkerboard_dist = std::numeric_limits<float>::max();
    if (fabs(dir.y) > 1e-3)  {
        float d = -(orig.y + 4) / dir.y; // the checkerboard plane has equation y = -4
        Vec3f pt = orig + dir * d;
        if (d > 0 && fabs(pt.x) < 10000 && pt.z < 10000 && pt.z > -10000 &&
                d < spheres_dist && d < model_distance) {
            checkerboard_dist = d;
            hit = pt;
            N = Vec3f(0,1,0);
            material = (int(0.25 * hit.x + 1000) + int(0.25 * hit.z)) & 1 ?
                       Material(1.5, Vec4f(0.4,0.1,.8,1.5), Vec3f(0., 0., 0.),100.) :
                       Material(1.5, Vec4f(0.4,0.1,.8,1.5), Vec3f(1., 204./255., 204./255.),100.);

        }
    }
    return std::min(spheres_dist, std::min(checkerboard_dist, model_distance)) < 1000;
}


Vec3f cast_ray(const Ray &r, size_t depth = 0) {
    auto dir = r.direction;
    Vec3f point, N;
    Material material;
    auto x = dir.x;
    auto y = dir.y;
    auto z = dir.z;
    if (depth > 3 || !scene_intersect(r, point, N, material)) {
        float max = std::max(std::max(fabsf(x), fabsf(y)), fabsf(z));
        if (fabsf(x) == max) {
            y /= max;
            z /= max;
            if (x < 0.00f) {
                uint32_t u = uint32_t(cubemap_width * ((-z * 0.50f) + 0.50f));
                uint32_t v = uint32_t(cubemap_height * ((-y * 0.50f) + 0.50f));
                return Vec3f(cubemap_neg_x_raw[((cubemap_width - u) + v * cubemap_width) * cubemap_channels],
                                 cubemap_neg_x_raw[((cubemap_width - u) + v * cubemap_width) * cubemap_channels + 1],
                                 cubemap_neg_x_raw[((cubemap_width - u) + v * cubemap_width) * cubemap_channels + 2]) * (1 / 255.99f);
            } else {
                uint32_t u = uint32_t(cubemap_width * ((-z * 0.50f) + 0.50f));
                uint32_t v = uint32_t(cubemap_height * ((-y * 0.50f) + 0.50f));
                return Vec3f(cubemap_pos_x_raw[(u + v * cubemap_width) * cubemap_channels],
                                 cubemap_pos_x_raw[(u + v * cubemap_width) * cubemap_channels + 1],
                                 cubemap_pos_x_raw[(u + v * cubemap_width) * cubemap_channels + 2])* (1 / 255.99f);
            }
        } else if (fabsf(y) == max) {
            x /= max;
            z /= max;
            if (y < 0.00f) {
                uint32_t u = uint32_t(cubemap_width * ((x * 0.50f) + 0.50f));
                uint32_t v = uint32_t(cubemap_height * ((-z * 0.50f) + 0.50f));
                return Vec3f(cubemap_neg_y_raw[(u + v * cubemap_width) * cubemap_channels],
                                 cubemap_neg_y_raw[(u + v * cubemap_width) * cubemap_channels + 1],
                                 cubemap_neg_y_raw[(u + v * cubemap_width) * cubemap_channels + 2]) * (1 / 255.99f);
            } else {
                uint32_t u = uint32_t(cubemap_width * ((x * 0.50f) + 0.50f));
                uint32_t v = uint32_t(cubemap_height * ((z * 0.50f) + 0.50f));
                return Vec3f(cubemap_pos_y_raw[(u + v * cubemap_width) * cubemap_channels],
                                 cubemap_pos_y_raw[(u + v * cubemap_width) * cubemap_channels + 1],
                                 cubemap_pos_y_raw[(u + v * cubemap_width) * cubemap_channels + 2]) * (1 / 255.99f);
            }
        } else {
            x /= max;
            y /= max;
            if (z < 0.00f) {
                uint32_t u = uint32_t(cubemap_width * ((-x * 0.50f) + 0.50f));
                uint32_t v = uint32_t(cubemap_height * ((-y * 0.50f) + 0.50f));
                return Vec3f(cubemap_neg_z_raw[(u + v * cubemap_width) * cubemap_channels],
                                 cubemap_neg_z_raw[(u + v * cubemap_width) * cubemap_channels + 1],
                                 cubemap_neg_z_raw[(u + v * cubemap_width) * cubemap_channels + 2]) * (1 / 255.99f);
            } else {
                uint32_t u = uint32_t(cubemap_width * ((-x * 0.50f) + 0.50f));
                uint32_t v = uint32_t(cubemap_height * ((-y * 0.50f) + 0.50f));
                return Vec3f(cubemap_pos_z_raw[((cubemap_width - u) + v * cubemap_width) * cubemap_channels],
                                 cubemap_pos_z_raw[((cubemap_width - u) + v * cubemap_width) * cubemap_channels + 1],
                                 cubemap_pos_z_raw[((cubemap_width - u) + v * cubemap_width) * cubemap_channels + 2]) * (1 / 255.99f);
            }
        }
    }
    Vec3f reflect_dir = reflect(dir, N).normalize();
    Vec3f refract_dir = refract(dir, N, material.refractive_index).normalize();
    // offset the original point to avoid occlusion by the object itself
    Vec3f reflect_orig = reflect_dir * N < 0 ? point - N * 1e-3 : point + N * 1e-3;
    Vec3f refract_orig = refract_dir * N < 0 ? point - N * 1e-3 : point + N * 1e-3;
    Vec3f reflect_color = cast_ray(Ray(reflect_orig, reflect_dir), depth + 1);
    Vec3f refract_color = cast_ray(Ray(refract_orig, refract_dir), depth + 1);
    float diffuse_light_intensity = 0, specular_light_intensity = 0;
    for (auto light : lights) {
        // I = sum(I_i * <N, light_dir>)
        Vec3f light_dir = (light.position - point).normalize();
        float light_distance = (light.position - point).norm();
        // checking if the point lies in the shadow of the lights[i]
        Vec3f shadow_orig = light_dir * N < 0 ? point - N * 1e-3 : point + N * 1e-3; // small shift of the point
        Vec3f shadow_pt, shadow_N;
        Material tmp_material;
        if (scene_intersect(Ray(shadow_orig, light_dir), shadow_pt, shadow_N, tmp_material) \
        && (shadow_pt - shadow_orig).norm() < light_distance)
            continue;
        diffuse_light_intensity += light.intensity * std::max(0.f, light_dir * N);
        specular_light_intensity += powf(std::max(0.f, -reflect(-light_dir, N) * dir), \
        material.specular_exponent) * light.intensity;
    }
    return material.diffuse_color * diffuse_light_intensity * material.albedo[0] + \
    Vec3f(1., 1., 1.) * specular_light_intensity * material.albedo[1] + \
    reflect_color * material.albedo[2] + refract_color * material.albedo[3];
}

// gaze direction, along the z axis, in the direction of minus infinity
// camera location, Vec3f (0,0,0)
void render() {
    const int width = 512;
    const int height = 512;
    const int fov = M_PI / 2.; // viewing angle
    std::vector<Vec3f> framebuffer(width * height);
    Vec3f orig = {0, 0, 0};
    boost::progress_display show_progress(framebuffer.size() * RAYS_PER_PIXEL);
    float jitter[8] = {
            -0.25f,  0.75f,
            0.75f,  0.25f,
            -0.75f, -0.25f,
            0.25f, -0.75f,
    };
#pragma omp parallel for
    for (uint32_t j = 0; j < height; ++j) {
        for (uint32_t i = 0; i < width; ++i) {
            framebuffer[i + j * width] = Vec3f(0, 0, 0);
            for (uint32_t k = 0; k < RAYS_PER_PIXEL; ++k) {
                float dir_x =  (i + 0.5 + jitter[2 * k]) -  width / 2.;
                float dir_y = -(j + 0.5 + jitter[2 * k + 1]) + height / 2.;
                float dir_z = -height / (2. * tan(fov / 2.));
                framebuffer[i + j * width] = framebuffer[i + j * width] +
                        cast_ray(Ray(orig, Vec3f(dir_x, dir_y, dir_z))) * (1 / float(RAYS_PER_PIXEL));
                ++show_progress;
            }
        }
    }

    //float x0 = (2.00f * (i + 0.50f + jitter[2 * i]) / width - 1) * tan(fov / 2.00f) * width / height;
    //float y0 = -(2.00f * (j + 0.50f + jitter[2 * i + 1]) / height - 1) * tan(fov / 2.00f);
//    for (size_t j = 0; j < height; j++) {
//        for (size_t i = 0; i < width; i++) {
//            float dir_x =  (i + 0.5) -  width / 2.;
//            float dir_y = -(j + 0.5) + height / 2.;    // this flips the image at the same time
//            float dir_z = -height / (2. * tan(fov / 2.));
//            Vec3f dir = Vec3f(dir_x, dir_y, dir_z).normalize();
//            framebuffer[i + j * width] = cast_ray(Ray(orig, dir));
//            ++show_progress;
//        }
//    }

    std::ofstream ofs; // save the framebuffer to file
    ofs.open("../out.ppm");
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (size_t i = 0; i < height * width; ++i) {
        Vec3f &c = framebuffer[i];
        float max = std::max(c[0], std::max(c[1], c[2]));
        if (max>1) c = c * (1. / max);
        for (size_t j = 0; j < 3; j++) {
            ofs << (char)(255 * std::max(0.f, std::min(1.f, framebuffer[i][j])));
        }
    }
    ofs.close();
}

int main() {
    cubemap_neg_x_raw = stbi_load("../Cubemaps/2/nx.png", &cubemap_width, &cubemap_height, &cubemap_channels, 0);
    cubemap_neg_y_raw = stbi_load("../Cubemaps/2/ny.png", &cubemap_width, &cubemap_height, &cubemap_channels, 0);
    cubemap_neg_z_raw = stbi_load("../Cubemaps/2/nz.png", &cubemap_width, &cubemap_height, &cubemap_channels, 0);
    cubemap_pos_x_raw = stbi_load("../Cubemaps/2/px.png", &cubemap_width, &cubemap_height, &cubemap_channels, 0);
    cubemap_pos_y_raw = stbi_load("../Cubemaps/2/py.png", &cubemap_width, &cubemap_height, &cubemap_channels, 0);
    cubemap_pos_z_raw = stbi_load("../Cubemaps/2/pz.png", &cubemap_width, &cubemap_height, &cubemap_channels, 0);
    Material pink_ivory(1.0, Vec4f(0.6, 0.3, 0.1, 0.0),Vec3f(191./255., 75./255.,  129./255.), 50.);
    Material glass(1.5, Vec4f(0.0, 0.5, 0.1, 0.8),Vec3f(0.6, 0.7, 0.8), 125.);
    Material red_rubber(1.0, Vec4f(0.9, 0.1, 0.0, 0.0),Vec3f(100./255., 10./255., 70./255.), 10.);
    Material yellow_ivory(1.0, Vec4f(0.6, 0.3, 0.1, 0.0),Vec3f(0.5, 0.4, 0.4), 50.);

    models.emplace_back("../teapot_2k_faces.obj", yellow_ivory, Vec3f(1, -4, -16));

    spheres.emplace_back(Vec3f(-4, -2.5, -15), 1.5, pink_ivory);
    spheres.emplace_back(Vec3f(-2, -3, -11),1, glass);
    spheres.emplace_back(Vec3f(6, -3, -16),1, red_rubber);

    lights.emplace_back(Vec3f(-30, 30, 10), 1.0);
    lights.emplace_back(Vec3f(-20, 40, 5), 1.2);

    render();
    return 0;
}