#include "HTTPClient.h"
#include <QEventLoop>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>

HTTPClient::HTTPClient() {
}

void HTTPClient::setUrl(const QString& url) {
	this->_url = url;
}

bool HTTPClient::request() {
	QEventLoop eventLoop;
 
    // "quit()" the event-loop, when the network request "finished()"
    QNetworkAccessManager mgr;
    QObject::connect(&mgr, SIGNAL(finished(QNetworkReply*)), &eventLoop, SLOT(quit()));
 
    // the HTTP request
	QNetworkRequest req(QUrl(_url.toUtf8().data()));
    QNetworkReply *re = mgr.get(req);
    eventLoop.exec(); // blocks stack until "finished()" has been called
 
    if (re->error() == QNetworkReply::NoError) {
		_reply = re->readAll();
		delete re;
		return true;
    } else {
		_error = re->errorString();
		delete re;
		return false;
    }
}

QString HTTPClient::reply() {
	return _reply;
}

QString HTTPClient::error() {
	return _error;
}
