#include "HumanComputation.h"
#include "Util.h"
#include "JSON.h"
#include "GradientDescent.h"
#include "MCMCUtil.h"

HumanComputation::HumanComputation() {
}


HumanComputation::~HumanComputation() {
}

/**
 * Human Computationのサーバデータベースの内容を初期化する。
 *
 * @param max_round		max round
 * @param max_step		max step
 */
void HumanComputation::init(int max_round, int max_step) {
	this->max_round = max_round;
	this->max_step = max_step;

	// HC初期化
	QString url = QString("http://gnishida.site90.com/config.php?current_round=0&max_round=%1&max_step=%2").arg(max_round).arg(max_step);
	client.setUrl(url);
	if (!client.request()) {
		throw client.reply();
		return;
	}
}

/**
 * サーバから、max_stepの値を取得する。
 * なぜ、わざわざサーバから取得するのか？　⇒クライアントを再起動するかもしれないから。
 *
 * @return		max_step
 */
int HumanComputation::getMaxStep() {
	client.setUrl("http://gnishida.site90.com/get_max_step.php");
	if (!client.request()) {
		throw client.reply();
	}

	return client.reply().toInt();
}

int HumanComputation::getCurrentRound() {
	client.setUrl("http://gnishida.site90.com/get_current_round.php");
	if (!client.request()) {
		throw client.reply();
	}
	return client.reply().toInt();
}

/**
 * サーバへ、タスクリストをアップロードする。
 * tasks[何番目のステップ]<オプション１の距離、オプション２の距離>
 *
 */
void HumanComputation::uploadTasks(vector<pair<vector<float>, vector<float> > >& tasks) {
	// HCタスクをアップロード
	for (int step = 0; step < max_step; ++step) {
		QString option1 = Util::join(tasks[step].first, ",");
		QString option2 = Util::join(tasks[step].second, ",");
		QString url = QString("http://gnishida.site90.com/add_task.php?step=%1&option1=%2&option2=%3").arg(step + 1).arg(option1).arg(option2);
		client.setUrl(url);
		if (!client.request()) {
			throw client.reply();
		}
	}
}

vector<pair<QString, QString> > HumanComputation::getTasks() {
	// feature一覧を取得
	client.setUrl("http://gnishida.site90.com/tasks.php");
	if (!client.request()) {
		throw client.reply();
	}

	return JSON::parse(client.reply(), "tasks", "option1", "option2");
}

vector<pair<QString, QString> > HumanComputation::getResults() {
	client.setUrl("http://gnishida.site90.com/results.php");
	if (!client.request()) {
		throw client.reply();
	}

	return JSON::parse(client.reply(), "results", "user_id", "choices");
}

void HumanComputation::uploadImage(const QString& filename) {
	client.setUrl("http://gnishida.site90.com/upload.php");
	if (!client.uploadFile("upload.php", "file", filename, "image/png")) {
		throw client.reply();
	}
}

void HumanComputation::nextRound() {
	client.setUrl("http://gnishida.site90.com/next_round.php");
	if (client.request()) {
		throw client.reply();
	}
}

QMap<int, vector<float> > HumanComputation::computePreferences() {
	std::vector<std::pair<QString, QString> > tasks = getTasks();

	// HC結果を取得
	std::vector<std::pair<QString, QString> > results = getResults();

	QMap<int, std::vector<float> > preferences;
	for (int u = 0; u < results.size(); ++u) {
		int user_id = results[u].first.toInt();
		QStringList chioces_list = results[u].second.split(",");

		GradientDescent gd;
		std::vector<std::pair<std::vector<float>, std::vector<float> > > features;
		std::vector<int> choices;

		for (int step = 0; step < tasks.size(); ++step) {
			std::vector<float> f1;
			std::vector<float> f2;
			QStringList feature1_list = tasks[step].first.split(",");
			QStringList feature2_list = tasks[step].second.split(",");
			for (int k = 0; k < 5; ++k) {
				f1.push_back(mcmcutil::MCMCUtil::distToFeature(64, feature1_list[k].toFloat()));
				f2.push_back(mcmcutil::MCMCUtil::distToFeature(64, feature2_list[k].toFloat()));
			}

			features.push_back(std::make_pair(f1, f2));

			choices.push_back(chioces_list[step].toInt() == 1 ? 1 : 0);
		}

		std::vector<float> w(8);
		w[0] = 0.3f; w[1] = 0.3f; w[2] = 0.3f; w[3] = 0.3f; w[4] = 0.3f; w[5] = 0.3f; w[6] = -0.3f; w[7] = -0.3f;
		gd.run(w, features, choices, 10000, false, 0.0, 0.001, 0.0001);
		
		preferences[user_id] = w;

		/*
		printf("User: %d: ", user_ids[u]);
		for (int k = 0; k < w.size(); ++k) {
			printf("%lf,", w[k]);
		}
		printf("\n");
		*/
	}

	return preferences;
}
