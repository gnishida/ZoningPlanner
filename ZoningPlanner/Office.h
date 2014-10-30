#pragma once

#include <QVector2D>

class Office
{
public:
	QVector2D location;

public:
	Office(const QVector2D& location) : location(location) {}
};

