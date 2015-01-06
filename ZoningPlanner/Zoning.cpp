#include "Zoning.h"
#include <QFile>
#include <QDomDocument>
#include "Util.h"
#include "BlockSet.h"

Zoning::Zoning() {
	// residential
	zoneTypeDistribution.push_back(0.2f);
	zoneTypeDistribution.push_back(0.38f);
	zoneTypeDistribution.push_back(0.2f);

	// commercial
	zoneTypeDistribution.push_back(0.06f);
	zoneTypeDistribution.push_back(0.05f);
	zoneTypeDistribution.push_back(0.03f);

	// manufacturing
	zoneTypeDistribution.push_back(0.02f);
	zoneTypeDistribution.push_back(0.01f);
	zoneTypeDistribution.push_back(0.01f);

	// park
	zoneTypeDistribution.push_back(0.02f);
	zoneTypeDistribution.push_back(0);
	zoneTypeDistribution.push_back(0);

	// amusement
	zoneTypeDistribution.push_back(0.01f);
	zoneTypeDistribution.push_back(0);
	zoneTypeDistribution.push_back(0);

	// public
	zoneTypeDistribution.push_back(0.01f);
	zoneTypeDistribution.push_back(0);
	zoneTypeDistribution.push_back(0);
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

/*int Zoning::getZone(const QVector2D& pt) const {
	int zoneId = -1;

	int cell_len = city_size / zone_size;

	int c = (pt.x() + cell_len * 0.5) / cell_len;
	int r = (pt.y() + cell_len * 0.5) / cell_len;
	int s = r * zone_size + c;

	return zones2[s];
}*/

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

			// to make level between [1,3]
			if (level < 1 || level > 3) level = 1;

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
 * 座標を、zones2のインデックス番号に変換する。
 *
 * @param city_length		cityの一辺の長さ
 * @param pt				座標
 * @return					インデックス番号
 */
int Zoning::positionToIndex(int city_length, const QVector2D& pt) const {
	int cell_len = city_length / zone_size;

	int c = (pt.x() + city_length * 0.5 + cell_len * 0.5) / cell_len;
	if (c < 0) c = 0;
	if (c >= zone_size) c = zone_size;

	int r = (pt.y() + city_length * 0.5 + cell_len * 0.5) / cell_len;
	if (r < 0) r = 0;
	if (r >= zone_size) r = zone_size;

	return r * zone_size + c;
}

