#pragma once

#include <QVector2D>

class Person
{
public:
	static enum { TYPE_UNKNOWN = 0, TYPE_STUDENT, TYPE_HOUSEWIFE, TYPE_OFFICEWORKER, TYPE_ELDERLY };

public:
	int type;
	QVector2D homeLocation;
	int commuteTo;
	int nearestStore;
	int nearestRestaurant;
	int nearestPark;
	int nearestAmusement;

public:
	Person() : type(TYPE_UNKNOWN), commuteTo(-1) {}
	Person(int type, const QVector2D& homeLocation) : type(type), homeLocation(homeLocation), commuteTo(-1) {}
};

