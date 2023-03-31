#version 330 core
layout (location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

out vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

void main()
{
  mat4 mvp = proj * view * model;
  gl_Position =  mvp * vec4(aPos.x, aPos.y, aPos.z, 1.0);
  normal = normalize(vec3(model*vec4(aNormal,0)));
}

