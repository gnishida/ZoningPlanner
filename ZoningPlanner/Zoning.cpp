#include "Zoning.h"

void ZoneType::init() {
	park_percentage = 0.05;
	parcel_area_mean = 2000;
	parcel_area_min = 1500;
	parcel_area_deviation = 1;
	parcel_split_deviation = 0.2;
	parcel_setback_front = 12;
	parcel_setback_sides = 5;
	parcel_setback_rear = 8;
	building_stories_mean = 2;
	building_stories_deviation = 50;
	building_max_depth = 0;
	building_max_frontage = 0;
	sidewalk_width = 4;
	tree_setback = 1;
	building_type = 0;
}

Zoning::Zoning() {
	Polygon2D polygon;
	polygon.push_back(QVector2D(-5000, -5000));
	polygon.push_back(QVector2D(5000, -5000));
	polygon.push_back(QVector2D(5000, 5000));
	polygon.push_back(QVector2D(-5000, 5000));
	zones.push_back(std::make_pair(polygon, ZoneType(ZoneType::TYPE_RESIDENTIAL, 1)));

	polygon.clear();
	polygon.push_back(QVector2D(-200, -200));
	polygon.push_back(QVector2D(200, -200));
	polygon.push_back(QVector2D(200, 200));
	polygon.push_back(QVector2D(-200, 200));
	zones.push_back(std::make_pair(polygon, ZoneType(ZoneType::TYPE_COMMERCIAL, 1)));
}