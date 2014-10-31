#include "Zoning.h"
#include <QFile>
#include <QDomDocument>

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
/*	Polygon2D polygon;
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
	*/
}

void Zoning::load(const QString& filename) {
	zones.clear();

	QFile file(filename);

	QDomDocument doc;
	doc.setContent(&file, true);
	QDomElement root = doc.documentElement();

	QDomNode node = root.firstChild();
	while (!node.isNull()) {
		if (node.toElement().tagName() == "zone") {
			int type = node.toElement().attribute("type").toInt();
			int level = node.toElement().attribute("level").toInt();
			ZoneType zone(type, level);

			Polygon2D polygon;
			QDomNode polygonNode = node.childNodes().at(0);
			for (int i = 0; i < polygonNode.childNodes().size(); ++i) {
				QDomNode pointNode = polygonNode.childNodes().at(i);
				float x = pointNode.toElement().attribute("x").toFloat();
				float y = pointNode.toElement().attribute("y").toFloat();
				polygon.push_back(QVector2D(x, y));
			}

			zones.push_back(std::make_pair(polygon, zone));
		}

		node = node.nextSibling();
	}
}