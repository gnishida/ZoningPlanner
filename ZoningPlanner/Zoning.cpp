#include "Zoning.h"
#include <QFile>
#include <QDomDocument>
#include "Util.h"
#include "BlockSet.h"

Zoning::Zoning() {
	zones = 0;
}

Zoning::~Zoning() {
	if (zones != 0) delete [] zones;
}

ZoneType Zoning::getZone(const QVector2D& pt) const {
	int s = positionToIndex(pt);
	if (s >= 0) {
		return ZoneType(zones[s], 1);
	} else {
		return ZoneType(ZoneType::TYPE_UNDEFINED, 1);
	}
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
			float y = nodePoint.toElement().attribute("y").toFloat();
			polygon.push_back(QVector2D(x, y));

			nodePoint = nodePoint.nextSibling();
		}

		init_zones.push_back(std::make_pair(polygon, zone));

		node = node.nextSibling();
	}
}

/**
 * xmlノードからゾーンデータを読み込む。
 * ゾーンはグリッド型で、(0,0)->(0,N),(1,0)->(1,N),..の順にxmlノードに格納されているとみなす。
 * また、レベルは現在1固定。
 */
void Zoning::load(const QString& filename) {
	// メモリ解放
	if (zones != 0) {
		delete [] zones;
		zones = 0;
	}

	QFile file(filename); 
    if (!file.open(QFile::ReadOnly| QFile::Truncate)) return;

	QDomDocument doc;
	int errorLine;
	QString errorStr;
    int errorColumn;
	doc.setContent(&file);
	QDomElement root = doc.documentElement();

	city_length = root.toElement().attribute("city_length").toInt();
	zone_size = root.toElement().attribute("zone_size").toInt();
	zones = new int[zone_size * zone_size];

	int count = 0;
	QDomNode nodeZone = root.firstChild();
	while (!nodeZone.isNull()) {
		int type = nodeZone.toElement().attribute("type").toInt();
		int level = nodeZone.toElement().attribute("level").toInt();
		int row = nodeZone.toElement().attribute("row").toInt();
		int col = nodeZone.toElement().attribute("col").toInt();

		zones[count++] = type;

		nodeZone = nodeZone.nextSibling();
	}
}

void Zoning::save(const QString& filename) {
	QFile file(filename);
    if (!file.open(QFile::WriteOnly| QFile::Truncate)) return;

	QDomDocument doc;
	QDomElement root = doc.createElement("zoning");
	root.setAttribute("city_length", city_length);
	root.setAttribute("zone_size", zone_size);
	doc.appendChild(root);

	for (int r = 0; r < zone_size; ++r) {
		for (int c = 0; c < zone_size; ++c) {
			QDomElement zoneNode = doc.createElement("zone");
			zoneNode.setAttribute("type", zones[r * zone_size + c]);
			zoneNode.setAttribute("level", 1);
			zoneNode.setAttribute("row", r);
			zoneNode.setAttribute("col", c);

			root.appendChild(zoneNode);
		}
	}

	QTextStream out(&file);
	doc.save(out, 4);
}

/**
 * 座標を、zonesのインデックス番号に変換する。
 *
 * @param city_length		cityの一辺の長さ
 * @param pt				座標
 * @return					インデックス番号
 */
int Zoning::positionToIndex(const QVector2D& pt) const {
	if (zones == 0) return -1;

	int cell_len = city_length / zone_size;

	int c = (pt.x() + city_length * 0.5 + cell_len * 0.5) / cell_len;
	if (c < 0) c = 0;
	if (c >= zone_size) c = zone_size;

	int r = (pt.y() + city_length * 0.5 + cell_len * 0.5) / cell_len;
	if (r < 0) r = 0;
	if (r >= zone_size) r = zone_size;

	return r * zone_size + c;
}

/**
 * zonesのインデックス番号を座標に変換する。
 */
QVector2D Zoning::indexToPosition(int index) const {
	int cell_len = city_length / zone_size;

	int c = index % zone_size;
	int r = index / zone_size;

	return QVector2D(((float)c + 0.5) * cell_len - city_length * 0.5, ((float)r + 0.5) * cell_len - city_length * 0.5);
}
