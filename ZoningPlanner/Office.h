#pragma once

#include <QVector2D>

class Office
{
public:
	static enum { TYPE_STORE = 0, TYPE_SCHOOL, TYPE_RESTAURANT, TYPE_PARK, TYPE_AMUSEMENT, TYPE_LIBRARY, TYPE_FACTORY, TYPE_STATION };

public:
	int type;
	QVector2D location;
	int level;
	float num;

public:
	Office(const QVector2D& location, int level) : location(location), level(level), num(0) {}
	Office(const QVector2D& location, int level, float num) : location(location), level(level), num(num) {}
};

