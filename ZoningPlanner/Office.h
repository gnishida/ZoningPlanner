#pragma once

#include <QVector2D>

class Office
{
public:
	QVector2D location;
	int level;

public:
	Office(const QVector2D& location, int level) : location(location), level(level) {}
};

