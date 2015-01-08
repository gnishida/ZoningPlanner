#pragma once

#include <QMap>
#include <QString>
#include "Polygon2D.h"
#include "BlockSet.h"
#include "ZoneType.h"

class Zoning {
public:
	int city_length;
	int* zones;
	int zone_size;

	std::vector<std::pair<Polygon2D, ZoneType> > init_zones;

public:
	Zoning();
	int getZone(const QVector2D& pt) const;
	void loadInitZones(const QString& filename);
	void load(QDomNode& node);
	//void save(const QString& filename);
	int positionToIndex(const QVector2D& pt) const;
	QVector2D indexToPosition(int index) const;
};

class CompareZoning
{
public:
    // Compare two Foo structs.
    bool operator()(const std::pair<float, Zoning>& x, const std::pair<float, Zoning>& y) const
    {
        return x.first < y.first;
    }
};

class CompareZoningReverse
{
public:
    // Compare two Foo structs.
    bool operator()(const std::pair<float, Zoning>& x, const std::pair<float, Zoning>& y) const
    {
        return x.first > y.first;
    }
};