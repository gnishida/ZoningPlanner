#pragma once

#include <QVector2D>
#include <vector>

class Person
{
public:
	static enum { TYPE_STUDENT = 0, TYPE_HOUSEWIFE, TYPE_OFFICEWORKER, TYPE_ELDERLY };

private:
	int _type;

public:
	QVector2D homeLocation;
	int nearestStore;
	int nearestSchool;
	int nearestRestaurant;
	int nearestPark;
	int nearestAmusement;
	int nearestLibrary;

	std::vector<float> preference;
	std::vector<float> feature;
	float score;

public:
	Person();
	Person(int type, const QVector2D& homeLocation);

	void setType(int type);
	int type() const { return _type; }
	void initPreference();
};

