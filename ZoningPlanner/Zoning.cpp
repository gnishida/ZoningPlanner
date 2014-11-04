#include "Zoning.h"
#include <QFile>
#include <QDomDocument>
#include "Util.h"

void ZoneType::init() {
	if (_type == TYPE_RESIDENTIAL) {
		if (_level == 1) {
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
		} else if (_level == 2) {
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
		} else if (_level == 3) {
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
	} else if (_type == TYPE_COMMERCIAL) {
		if (_level == 1) {
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
		} else if (_level == 2) {
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
		} else if (_level == 3) {
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
	} else if (_type == TYPE_MANUFACTURING) {
		if (_level == 1) {
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
		} else if (_level == 2) {
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
		} else if (_level == 3) {
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
	} else if (_type == TYPE_PARK) {
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
	} else if (_type == TYPE_AMUSEMENT) {
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
	} else if (_type == TYPE_PUBLIC) {
		park_percentage = 0.0f;
		parcel_area_mean = 2000;
		parcel_area_min = 1000;
		parcel_area_deviation = 1;
		parcel_split_deviation = 0.2;
		parcel_setback_front = 8;
		parcel_setback_sides = 6;
		parcel_setback_rear = 8;
		building_stories_mean = 3;
		building_stories_deviation = 50;
		building_max_depth = 0;
		building_max_frontage = 0;
		sidewalk_width = 4;
		tree_setback = 1;
		building_type = 0;
	}
}

void ZoneType::setType(int type) {
	_type = type;
	init();
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

	{
		Polygon2D polygon;
		polygon.push_back(QVector2D(-100000, -100000));
		polygon.push_back(QVector2D(100000, -100000));
		polygon.push_back(QVector2D(100000, 100000));
		polygon.push_back(QVector2D(-100000, 100000));
		zones.push_back(std::make_pair(polygon, ZoneType(ZoneType::TYPE_RESIDENTIAL, 1)));
	}

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

void Zoning::save(const QString& filename) {
	QFile file(filename);
     
    if (!file.open(QFile::WriteOnly| QFile::Truncate)) return;

	QDomDocument doc;
	QDomElement root = doc.createElement("zoning");
	doc.appendChild(root);

	for (int i = 0; i < zones.size(); ++i) {
		QDomElement zoneNode = doc.createElement("zone");
		zoneNode.setAttribute("type", zones[i].second.type());
		zoneNode.setAttribute("level", zones[i].second.level());
		
		QDomElement polygonNode = doc.createElement("polygon");
		for (int j = 0; j < zones[i].first.size(); ++j) {
			QDomElement pointNode = doc.createElement("point");
			pointNode.setAttribute("x", zones[i].first.at(j).x());
			pointNode.setAttribute("y", zones[i].first.at(j).y());
			polygonNode.appendChild(pointNode);
		}
		zoneNode.appendChild(polygonNode);

		root.appendChild(zoneNode);
	}

	QTextStream out(&file);
	doc.save(out, 4);
}

/**
 * Randomly generate zoning plan.
 */
void Zoning::generate(const QVector2D& size) {
	zones.clear();

	{
		Polygon2D polygon;
		polygon.push_back(QVector2D(-100000, -100000));
		polygon.push_back(QVector2D(100000, -100000));
		polygon.push_back(QVector2D(100000, 100000));
		polygon.push_back(QVector2D(-100000, 100000));
		zones.push_back(std::make_pair(polygon, ZoneType(ZoneType::TYPE_RESIDENTIAL, 1)));
	}

	float step = 200.0f;
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
			int r = Util::genRand(0, 12);
			if (r <= 2) {
				zones.push_back(std::make_pair(polygon, ZoneType(ZoneType::TYPE_RESIDENTIAL, r+1)));
			} else if (r <= 5) {
				zones.push_back(std::make_pair(polygon, ZoneType(ZoneType::TYPE_COMMERCIAL, r-2)));
			} else if (r <= 8) {
				zones.push_back(std::make_pair(polygon, ZoneType(ZoneType::TYPE_MANUFACTURING, r-5)));
			} else if (r == 9) {
				zones.push_back(std::make_pair(polygon, ZoneType(ZoneType::TYPE_PARK, 1)));
			} else if (r == 10) {
				zones.push_back(std::make_pair(polygon, ZoneType(ZoneType::TYPE_AMUSEMENT, 1)));
			} else if (r == 11) {
				zones.push_back(std::make_pair(polygon, ZoneType(ZoneType::TYPE_PUBLIC, 1)));
			}			
		}
	}
}