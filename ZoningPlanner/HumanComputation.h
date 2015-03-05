#pragma once

#include <vector>
#include <QString>
#include "HTTPClient.h"
#include <QMap>

using namespace std;

class HumanComputation {
private:
	HTTPClient client;
	int max_round;
	int max_step;

public:
	HumanComputation(void);
	~HumanComputation(void);

	void init(int max_round, int max_step);
	int getMaxStep();
	int getCurrentRound();
	void uploadTasks(vector<pair<vector<float>, vector<float> > >& tasks);
	vector<pair<QString, QString> > getTasks();
	vector<pair<QString, QString> > getResults();
	void uploadImage(const QString& filename);
	void nextRound();
	QMap<int, vector<float> > computePreferences();
};

