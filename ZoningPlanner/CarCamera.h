#pragma once

#include "Camera.h"
#include "VBORenderManager.h"

class CarCamera : public Camera {
public:
	VBORenderManager* rendManager;

	float carHeight;
	float theta;
	QVector3D pos;
	QVector3D viewDir;
	QVector3D up;

public:
	CarCamera() : carHeight(2.0f), theta(0.0f) { fovy = 60.0f; type = TYPE_CAR; }

	void updatePerspective(int width,int height);
	void updateCamMatrix();
	void resetCamera();
	void moveForward(float speed);
	void steer(float th);
	void saveCameraPose(const QString &filepath);
	void loadCameraPose(const QString &filepath);
};

