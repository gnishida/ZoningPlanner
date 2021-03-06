#include "Person.h"
#include "Util.h"

Person::Person() : _type(TYPE_STUDENT) {
	initPreference();
}

Person::Person(int type, const QVector2D& homeLocation) : _type(type), homeLocation(homeLocation) {
	initPreference();
}

void Person::setType(int type) {
	_type = type;
	initPreference();
}

/**
 * Define the preference vector.
 * Each component is 
 *    distance to a store
 *    distance to a school
 *    distance to a restaurant
 *    distance to a park
 *    distance to a amusement
 *    distance to a library
 *    noise
 *    air pollution
 *    distance to a station
 */
void Person::initPreference() {
	float r = Util::genRand(0, 1);

	preference.resize(8);

	if (_type == TYPE_STUDENT) {
		if (r < 0.333) { // outdoor life
			preference[0] = 0; preference[1] = 0; preference[2] = 0.15; preference[3] = 0.15; preference[4] = 0.3; preference[5] = 0; preference[6] = 0.1; preference[7] = 0.1;
		} else if (r < 0.666) { // indoor life
			preference[0] = 0; preference[1] = 0; preference[2] = 0.15; preference[3] = 0; preference[4] = 0.55; preference[5] = 0; preference[6] = 0.2; preference[7] = 0.1;
		} else { // study hard
			preference[0] = 0; preference[1] = 0; preference[2] = 0.05; preference[3] = 0; preference[4] = 0; preference[5] = 0; preference[6] = 0.25; preference[7] = 0.1;
		}
	} else if (_type == TYPE_HOUSEWIFE) {
		if (r < 0.7) { // with kids
			preference[0] = 0.18; preference[1] = 0.17; preference[2] = 0; preference[3] = 0.17; preference[4] = 0; preference[5] = 0.08; preference[6] = 0.2; preference[7] = 0.2;
		} else { // no kid
			preference[0] = 0.3; preference[1] = 0; preference[2] = 0.3; preference[3] = 0.1; preference[4] = 0; preference[5] = 0; preference[6] = 0.1; preference[7] = 0.2;
		}
	} else if (_type == TYPE_OFFICEWORKER) {
		if (r < 0.3) { // men with kids
			preference[0] = 0.05; preference[1] = 0; preference[2] = 0.1; preference[3] = 0.2; preference[4] = 0.1; preference[5] = 0; preference[6] = 0.1; preference[7] = 0.15;
		} else if (r < 0.6) { // women with kids
			preference[0] = 0.15; preference[1] = 0.1; preference[2] = 0; preference[3] = 0.15; preference[4] = 0; preference[5] = 0.1; preference[6] = 0.1; preference[7] = 0.2;
		} else { // no kid
			preference[0] = 0.2; preference[1] = 0; preference[2] = 0.25; preference[3] = 0; preference[4] = 0.15; preference[5] = 0; preference[6] = 0.1; preference[7] = 0.1;
		}
	} else if (_type == TYPE_ELDERLY) {
		if (r < 0.5) { // with kids family
			preference[0] = 0.3; preference[1] = 0; preference[2] = 0.15; preference[3] = 0.05; preference[4] = 0; preference[5] = 0; preference[6] = 0.25; preference[7] = 0.25;
		} else { // live alone
			preference[0] = 0.4; preference[1] = 0; preference[2] = 0.2; preference[3] = 0; preference[4] = 0; preference[5] = 0; preference[6] = 0.2; preference[7] = 0.2;
		}
	}


	float total = 0.0f;
	for (int i = 0; i < preference.size(); ++i) {
		total += preference[i];
	}
	if (fabs(total - 1.0f) > 0.01f) {
		printf("ERR: Some preference vector are not normalized!!\n");
	}
}
