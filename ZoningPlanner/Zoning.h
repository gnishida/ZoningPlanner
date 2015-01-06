#pragma once

#include <QMap>
#include <QString>
#include "Polygon2D.h"
#include "BlockSet.h"
#include "ZoneType.h"

class Zoning {
public:
	int city_size;
	std::vector<std::pair<Polygon2D, ZoneType> > zones;
	int* zones2;
	int zone_size;

	// 改善案
	std::vector<float> zoneTypeDistribution;

public:
	Zoning();
	size_t size() { return zones.size(); }
	int getZone(const QVector2D& pt) const;
	void load(const QString& filename);
	void save(const QString& filename);
	int positionToIndex(int city_length, const QVector2D& pt) const;
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