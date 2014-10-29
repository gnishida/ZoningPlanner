#pragma once

#include <QMap>
#include "Polygon2D.h"

class ZoneType
{
public:
	static enum { TYPE_UNKNOWN = 0, TYPE_RESIDENTIAL, TYPE_COMMERCIAL, TYPE_MANUFACTURING, TYPE_PARK, TYPE_AMUSEMENT };

public:
	ZoneType() : type(TYPE_UNKNOWN), level(0) { init(); }
	ZoneType(int type, int level) : type(type), level(level) { init(); }
	void init();

public:
	int type;
	int level;

	float park_percentage;
	float parcel_area_mean;
	float parcel_area_min;
	float parcel_area_deviation;
	float parcel_split_deviation;
	float parcel_setback_front;
	float parcel_setback_sides;
	float parcel_setback_rear;
	float building_stories_mean;
	float building_stories_deviation;
	float building_max_depth;
	float building_max_frontage;
	float sidewalk_width;
	float tree_setback;
	int building_type;
};

class Zoning {
public:
	/*std::vector<Polygon2D> polygons;
	std::vector<ZoneType> zones;*/
	std::vector<std::pair<Polygon2D, ZoneType> > zones;
	//QMap<Polygon2D, ZoneType> zones;

public:
	Zoning();
	size_t size() { return zones.size(); }
};