#pragma once

#include <QString>

class HTTPClient {
private:
	QString _url;
	QString _reply;

public:
	HTTPClient();
	void setUrl(const QString& url);
	bool request();
	bool uploadFile(const QString& action, const QString& form_name, const QString& filename, const QString& filetype);
	QString reply();
};

