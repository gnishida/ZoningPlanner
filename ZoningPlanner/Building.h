#pragma once

#include <QSettings>
#include "Polygon3D.h"
#include <QColor>

class Building {
public:
	Polygon3D footprint;
	int bldType;
	int subType;
	int numStories;
	QColor color;
	int roofTextureId;

public:
	Building() : bldType(-1), numStories(-1) {}
};
