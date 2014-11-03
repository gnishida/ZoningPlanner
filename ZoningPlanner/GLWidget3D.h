/*********************************************************************
This file is part of QtUrban.

    QtUrban is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3 of the License.

    QtUrban is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with QtUrban.  If not, see <http://www.gnu.org/licenses/>.
***********************************************************************/

#pragma once

#include "glew.h"
#include "VBORenderManager.h"
#include "GLWidget3D_Shadows.h"

#include <QGLWidget>
#include <QMouseEvent>
#include <QKeyEvent>
#include "Camera.h"
#include "Camera2D.h"
//#include "CarCamera.h"
#include "RoadGraph.h"

class MainWindow;

class GLWidget3D : public QGLWidget {
public:
	MainWindow* mainWin;
	Camera2D camera2D;
	//CarCamera carCamera;
	Camera* camera;

	bool shiftPressed;
	bool controlPressed;
	bool altPressed;
	bool keyMPressed;
	QPoint lastPos;
	float farPlaneToSpaceRadiusFactor;
	float spaceRadius;
	float rotationSensitivity;
	float zoomSensitivity;

	VBORenderManager vboRenderManager;
	GLWidgetSimpleShadow shadow;

public:
	GLWidget3D(MainWindow *parent);
	~GLWidget3D() {}

	QSize minimumSizeHint() const;
	QSize sizeHint() const;
	void keyPressEvent(QKeyEvent* e);
	void keyReleaseEvent(QKeyEvent* e);

	void mouseTo2D(int x, int y, QVector2D &result);

	void drawScene(int drawMode);

	void updateCamera();
	//void generate3DGeometry(bool justRoads=false);

protected:
	void initializeGL();

	void resizeGL(int width, int height);
	void paintGL();    

	void mousePressEvent(QMouseEvent *e);
	void mouseMoveEvent(QMouseEvent *e);
	void mouseReleaseEvent(QMouseEvent *e);
};

