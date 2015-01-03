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
	QString reply();
	QString error();
};

