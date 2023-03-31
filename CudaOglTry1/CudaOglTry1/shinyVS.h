#version 330 core
layout (location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

out vec3 normal;
out vec3 viewVec;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform vec3 camPosW;

void main()
{
  mat4 mvp = proj * view * model;
  vec4 currP = vec4(aPos.x, aPos.y, aPos.z, 1.0);

  gl_Position = mvp * currP;
  
  normal = normalize(vec3(model*vec4(aNormal,0)));
  
  vec3 wp = vec3(model * currP);
  viewVec = normalize(wp - camPosW);
}

