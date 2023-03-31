#version 330 core

in vec3 normal;

uniform vec4 albedo;
uniform vec4 botSkyColor;
uniform vec4 topSkyColor;

out vec4 FragColor;

void main()
{
	vec3 t = vec3(0.5, 0.5, 0.5) * (normal + vec3(1.0, 1.0, 1.0));
	
	FragColor = ((1.0 - t.y) * botSkyColor + t.y * topSkyColor) * albedo ;
}

