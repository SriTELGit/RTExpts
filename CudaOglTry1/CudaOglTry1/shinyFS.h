#version 330 core

in vec3 normal;
in vec3 viewVec;

uniform vec4 albedo;
uniform vec4 botSkyColor;
uniform vec4 topSkyColor;
uniform float roughness;

out vec4 FragColor;

void main()
{
	vec3 reflVec = reflect(viewVec, normal);

	vec3 tR = vec3(0.5, 0.5, 0.5) * (reflVec + vec3(1.0, 1.0, 1.0));
	vec3 tL = vec3(0.5, 0.5, 0.5) * (normal + vec3(1.0, 1.0, 1.0));

	vec3 skyCR = ((1.0 - tR.y) * botSkyColor + tR.y * topSkyColor).rgb;
	vec3 skyCL = ((1.0 - tL.y) * botSkyColor + tL.y * topSkyColor).rgb;

	//FragColor = vec4(skyCR, 1.0); //testing reflection
	FragColor = vec4(skyCL * albedo.rgb*roughness + skyCR.rgb*(1.0-roughness), 1.0); //

}

