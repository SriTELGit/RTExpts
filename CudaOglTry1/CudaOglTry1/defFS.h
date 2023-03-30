#version 330 core

in vec3 color;
in vec2 tex;

uniform sampler2D texture0;

out vec4 FragColor;

void main()
{
  FragColor = vec4(color,1.0f) * texture(texture0, tex);
}

