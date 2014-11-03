#pragma once

#include <QVector2D>

class Person
{
public:
	static enum { TYPE_UNKNOWN = 0, TYPE_STUDENT, TYPE_HOUSEWIFE, TYPE_OFFICEWORKER, TYPE_ELDERLY };

private:
	int _type;

public:
	QVector2D homeLocation;
	int commuteTo;
	int nearestStore;
	int nearestRestaurant;
	int nearestPark;
	int nearestAmusement;
	float preference[8];

public:
	Person();
	Person(int type, const QVector2D& homeLocation);

	int type() { return _type; }
	void initPreference();
};

