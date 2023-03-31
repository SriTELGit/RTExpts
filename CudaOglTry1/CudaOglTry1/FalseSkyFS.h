#version 330 core


out vec4 FragColor;

in vec2 xypos;

uniform vec4 botSkyColor;
uniform vec4 topSkyColor;

void main()
{
	vec2 t = vec2(0.5,0.5) * (xypos + vec2(1.0,1.0));
	FragColor = (1.0 - t.y) * botSkyColor + t.y * topSkyColor;
}

