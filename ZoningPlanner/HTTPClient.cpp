#include "HTTPClient.h"
#include <QEventLoop>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QHttp>
#include <QFile>
#include <QString>
#include <QByteArray>
#include <iostream>

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
		int index = _reply.indexOf("\n");
		if (index >= 0) {
			_reply = _reply.mid(0, index);
		}

		delete re;
		return true;
    } else {
		_reply = re->errorString();
		delete re;
		return false;
    }
}

bool HTTPClient::uploadFile(const QString& action, const QString& form_name, const QString& filename, const QString& filetype) {
	QFile file(filename);
	if (!file.open(QIODevice::ReadOnly)) return false;

	QString boundary="-----------------------------7d935033608e2";

	QByteArray body(QString("--" + boundary + "\r\n").toAscii());
	body.append("Content-Disposition: form-data; name=\"action\"\r\n\r\n");
	body.append(action + "\r\n");
	body.append("--" + boundary + "\r\n");
	body.append("Content-Disposition: form-data; name=\"" + form_name + "\"; filename=\"" + filename + "\"\r\n");
	body.append("Content-Type: " + filetype + "\r\n\r\n");
	body.append(file.readAll());
	body.append("\r\n");
	body.append("--" + boundary + "--\r\n");


	QEventLoop eventLoop;
 
    QNetworkAccessManager mgr;
    QObject::connect(&mgr, SIGNAL(finished(QNetworkReply*)), &eventLoop, SLOT(quit()));
	QNetworkRequest req(QUrl(_url.toUtf8().data()));
	req.setRawHeader(QString("Content-Type").toAscii(), QString("multipart/form-data; boundary=" + boundary).toAscii());
	req.setRawHeader(QString("Content-Length").toAscii(), QString::number(body.length()).toAscii());

	QNetworkReply *re = mgr.post(req, body);
    eventLoop.exec();

    if (re->error() == QNetworkReply::NoError) {
		_reply = re->readAll();
		int index = _reply.indexOf("\n");
		if (index >= 0) {
			_reply = _reply.mid(0, index);
		}

		delete re;
		return true;
    } else {
		_reply = re->errorString();
		delete re;
		return false;
    }
}

QString HTTPClient::reply() {
	return _reply;
}

