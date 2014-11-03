#include "Zoning.h"
#include <QFile>
#include <QDomDocument>
#include "Util.h"

void ZoneType::init() {
	if (type == TYPE_RESIDENTIAL) {
		if (level == 1) {
			park_percentage = 0.0f;
			parcel_area_mean = 200;
			parcel_area_min = 100;
			parcel_area_deviation = 1;
			parcel_split_deviation = 0.2;
			parcel_setback_front = 2;
			parcel_setback_sides = 2;
			parcel_setback_rear = 2;
			building_stories_mean = 2;
			building_stories_deviation = 50;
			building_max_depth = 0;
			building_max_frontage = 0;
			sidewalk_width = 2;
			tree_setback = 1;
			building_type = 0;
		} else if (level == 2) {
			park_percentage = 0.0f;
			parcel_area_mean = 2000;
			parcel_area_min = 1000;
			parcel_area_deviation = 1;
			parcel_split_deviation = 0.2;
			parcel_setback_front = 8;
			parcel_setback_sides = 6;
			parcel_setback_rear = 8;
			building_stories_mean = 4;
			building_stories_deviation = 50;
			building_max_depth = 0;
			building_max_frontage = 0;
			sidewalk_width = 3;
			tree_setback = 1;
			building_type = 0;
		} else if (level == 3) {
			park_percentage = 0.0f;
			parcel_area_mean = 4000;
			parcel_area_min = 2000;
			parcel_area_deviation = 1;
			parcel_split_deviation = 0.2;
			parcel_setback_front = 10;
			parcel_setback_sides = 8;
			parcel_setback_rear = 12;
			building_stories_mean = 12;
			building_stories_deviation = 50;
			building_max_depth = 0;
			building_max_frontage = 0;
			sidewalk_width = 4;
			tree_setback = 1;
			building_type = 0;
		}
	} else if (type == TYPE_COMMERCIAL) {
		if (level == 1) {
			park_percentage = 0.0f;
			parcel_area_mean = 200;
			parcel_area_min = 100;
			parcel_area_deviation = 1;
			parcel_split_deviation = 0.2;
			parcel_setback_front = 2;
			parcel_setback_sides = 2;
			parcel_setback_rear = 2;
			building_stories_mean = 2;
			building_stories_deviation = 50;
			building_max_depth = 0;
			building_max_frontage = 0;
			sidewalk_width = 2;
			tree_setback = 1;
			building_type = 0;
		} else if (level == 2) {
			park_percentage = 0.0f;
			parcel_area_mean = 2000;
			parcel_area_min = 1000;
			parcel_area_deviation = 1;
			parcel_split_deviation = 0.2;
			parcel_setback_front = 8;
			parcel_setback_sides = 6;
			parcel_setback_rear = 8;
			building_stories_mean = 4;
			building_stories_deviation = 50;
			building_max_depth = 0;
			building_max_frontage = 0;
			sidewalk_width = 3;
			tree_setback = 1;
			building_type = 0;
		} else if (level == 3) {
			park_percentage = 0.0f;
			parcel_area_mean = 4000;
			parcel_area_min = 2000;
			parcel_area_deviation = 1;
			parcel_split_deviation = 0.2;
			parcel_setback_front = 10;
			parcel_setback_sides = 8;
			parcel_setback_rear = 12;
			building_stories_mean = 12;
			building_stories_deviation = 50;
			building_max_depth = 0;
			building_max_frontage = 0;
			sidewalk_width = 4;
			tree_setback = 1;
			building_type = 0;
		}
	} else if (type == TYPE_MANUFACTURING) {
		if (level == 1) {
			park_percentage = 0.05;
			parcel_area_mean = 200;
			parcel_area_min = 100;
			parcel_area_deviation = 1;
			parcel_split_deviation = 0.2;
			parcel_setback_front = 2;
			parcel_setback_sides = 2;
			parcel_setback_rear = 2;
			building_stories_mean = 2;
			building_stories_deviation = 50;
			building_max_depth = 0;
			building_max_frontage = 0;
			sidewalk_width = 2;
			tree_setback = 1;
			building_type = 0;
		} else if (level == 2) {
			park_percentage = 0.0f;
			parcel_area_mean = 2000;
			parcel_area_min = 1000;
			parcel_area_deviation = 1;
			parcel_split_deviation = 0.2;
			parcel_setback_front = 5;
			parcel_setback_sides = 4;
			parcel_setback_rear = 4;
			building_stories_mean = 4;
			building_stories_deviation = 50;
			building_max_depth = 0;
			building_max_frontage = 0;
			sidewalk_width = 3;
			tree_setback = 1;
			building_type = 0;
		} else if (level == 3) {
			park_percentage = 0.0f;
			parcel_area_mean = 4000;
			parcel_area_min = 2000;
			parcel_area_deviation = 1;
			parcel_split_deviation = 0.2;
			parcel_setback_front = 7;
			parcel_setback_sides = 4;
			parcel_setback_rear = 4;
			building_stories_mean = 12;
			building_stories_deviation = 50;
			building_max_depth = 0;
			building_max_frontage = 0;
			sidewalk_width = 4;
			tree_setback = 1;
			building_type = 0;
		}
	} else if (type == TYPE_PARK) {
		park_percentage = 1.0f;
		parcel_area_mean = 4000;
		parcel_area_min = 2000;
		parcel_area_deviation = 1;
		parcel_split_deviation = 0.2;
		parcel_setback_front = 7;
		parcel_setback_sides = 4;
		parcel_setback_rear = 4;
		building_stories_mean = 1;
		building_stories_deviation = 50;
		building_max_depth = 0;
		building_max_frontage = 0;
		sidewalk_width = 4;
		tree_setback = 1;
		building_type = 0;
	} else if (type == TYPE_AMUSEMENT) {
		park_percentage = 0.0f;
		parcel_area_mean = 2000;
		parcel_area_min = 1000;
		parcel_area_deviation = 1;
		parcel_split_deviation = 0.2;
		parcel_setback_front = 8;
		parcel_setback_sides = 6;
		parcel_setback_rear = 8;
		building_stories_mean = 4;
		building_stories_deviation = 50;
		building_max_depth = 0;
		building_max_frontage = 0;
		sidewalk_width = 4;
		tree_setback = 1;
		building_type = 0;
	}
}

Zoning::Zoning() {
}

int Zoning::getZone(const QVector2D& pt) const {
	int zoneId = -1;

	for (int i = 0; i < zones.size(); ++i) {
		if (zones[i].first.contains(QVector2D(pt))) {
			zoneId = i;
		}
	}

	return zoneId;
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

/**
 * Randomly generate zoning plan.
 */
void Zoning::generate(const QVector2D& size) {
	float step = 100.0f;
	for (int u = 0; u < size.x() / step; ++u) {
		float x = (float)u * step;
		for (int v = 0; v < size.y() / step; ++v) {
			float y = (float)v * step;

			Polygon2D polygon;
			polygon.push_back(QVector2D(x, y));
			polygon.push_back(QVector2D(x + step, y));
			polygon.push_back(QVector2D(x + step, y + step));
			polygon.push_back(QVector2D(x, y + step));

			ZoneType zone;
			int r = Util::genRand(0, 11);
			if (r <= 2) {
				zone.type = ZoneType::TYPE_RESIDENTIAL;
				zone.level = r + 1;
			} else if (r <= 5) {
				zone.type = ZoneType::TYPE_COMMERCIAL;
				zone.level = r - 2;
			} else if (r <= 8) {
				zone.type = ZoneType::TYPE_MANUFACTURING;
				zone.level = r - 5;
			} else if (r == 9) {
				zone.type = ZoneType::TYPE_PARK;
				zone.level = 1;
			} else if (r == 10) {
				zone.type = ZoneType::TYPE_AMUSEMENT;
				zone.level = 1;
			}

			zones.push_back(std::make_pair(polygon, zone));
		}
	}
}