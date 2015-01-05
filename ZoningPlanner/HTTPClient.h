#pragma once

#include <QString>

class HTTPClient {
private:
	QString _url;
	QString _reply;
	QString _error;

public:
	HTTPClient();
	void setUrl(const QString& url);
	bool request();
	void uploadFile(const QString& hostname, const QString& action, const QString& filename);
	QString reply();
	QString error();
};

