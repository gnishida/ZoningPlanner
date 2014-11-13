#pragma once

#include <QVector2D>

class Office
{
public:
	QVector2D location;
	int level;
	float num;

public:
	Office(const QVector2D& location, int level) : location(location), level(level), num(0) {}
	Office(const QVector2D& location, int level, float num) : location(location), level(level), num(num) {}
};

