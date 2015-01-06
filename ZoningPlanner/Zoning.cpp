#include "Zoning.h"
#include <QFile>
#include <QDomDocument>
#include "Util.h"
#include "BlockSet.h"

Zoning::Zoning() {
}

int Zoning::getZone(const QVector2D& pt) const {
	int s = positionToIndex(pt);
	return zones[s];
}

void Zoning::loadInitZones(const QString& filename) {
	init_zones.clear();

	QFile file(filename);

	QDomDocument doc;
	doc.setContent(&file, true);
	QDomElement root = doc.documentElement();
 
    QDomNode node = root.firstChild();
	while (!node.isNull()) {
		int type = node.toElement().attribute("type").toInt();
		int level = node.toElement().attribute("level").toInt();
		if (level < 1) level = 1;
		if (level > 3) level = 3;

		ZoneType zone(type, level);

		Polygon2D polygon;
		QDomNode nodePoint = node.firstChild();
		while (!nodePoint.isNull()) {
			float x = nodePoint.toElement().attribute("x").toFloat();
			float y = nodePoint.toElement().attribute("x").toFloat();
			polygon.push_back(QVector2D(x, y));

			nodePoint = nodePoint.nextSibling();
		}

		init_zones.push_back(std::make_pair(polygon, zone));

		node = node.nextSibling();
	}
}

void Zoning::load(QDomNode& node) {
	int startSize = node.toElement().attribute("startSize").toInt();
	int numLayers = node.toElement().attribute("numLayers").toInt();

	QDomNode nodeZone = node.firstChild();
	while (!nodeZone.isNull()) {
		int type = nodeZone.toElement().attribute("type").toInt();
		int level = nodeZone.toElement().attribute("level").toInt();

		nodeZone = nodeZone.nextSibling();
	}
}

/*void Zoning::save(const QString& filename) {
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
*/

/**
 * 座標を、zones2のインデックス番号に変換する。
 *
 * @param city_length		cityの一辺の長さ
 * @param pt				座標
 * @return					インデックス番号
 */
int Zoning::positionToIndex(const QVector2D& pt) const {
	int cell_len = city_length / zone_size;

	int c = (pt.x() + city_length * 0.5 + cell_len * 0.5) / cell_len;
	if (c < 0) c = 0;
	if (c >= zone_size) c = zone_size;

	int r = (pt.y() + city_length * 0.5 + cell_len * 0.5) / cell_len;
	if (r < 0) r = 0;
	if (r >= zone_size) r = zone_size;

	return r * zone_size + c;
}

