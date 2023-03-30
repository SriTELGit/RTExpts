#version 330 core
layout (location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec2 aTex;

out vec3 color;
out vec2 tex;


uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

void main()
{
  mat4 mvp = proj * view * model;
  gl_Position =  mvp * vec4(aPos.x, aPos.y, aPos.z, 1.0);
  color = aColor;
  tex = aTex;
}

