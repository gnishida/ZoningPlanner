#pragma once


#include "LC_Layer.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"



class VBORenderManager;

class VBOLayer {



public:
	VBOLayer();

	void init(VBORenderManager& rendManager);

	void render(VBORenderManager& rendManager);
	void updateTerrain(float coordX,float coordY,float change,float rad);
	void updateTerrainNewValue(float coordX,float coordY,float newValue,float rad);
	float getTerrainHeight(float xM,float yM);
	void loadTerrain(QString& fileName);
	void saveTerrain(QString& fileName);

	// edit
	Layer layer;
	bool initialized;
	int resolutionX;
	int resolutionY;

private:
	GLuint elementbuffer;
	GLuint vbo;
	GLuint indicesCount;
	GLuint program;
};