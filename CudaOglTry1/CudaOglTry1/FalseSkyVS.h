#version 330 core
layout (location = 0) in vec3 aPos;


out vec2 xypos;

void main()
{
  gl_Position =  vec4(aPos.x, aPos.y, 0.99, 1.0);
  xypos = vec2(aPos.x, aPos.y);
}

