#include "HTTPClient.h"
#include <QEventLoop>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QHttp>
#include <QFile>
#include <QString>
#include <QByteArray>

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
		_error = re->errorString();
		delete re;
		return false;
    }
}

bool HTTPClient::uploadFile(const QString& hostname, const QString& action, const QString& filename) {
	//QByteArray data;
	QFile file(filename);
	if (!file.open(QIODevice::ReadOnly))
		return false;

	/*
	QDataStream in(&file);
	in.setVersion(QDataStream::Qt_4_6);
	in >> data ;
	*/

	QString boundary="-----------------------------7d935033608e2";

	QByteArray body(QString("--" + boundary + "\r\n").toAscii());
	body.append("Content-Disposition: form-data; name=\"action\"\r\n\r\n");
	body.append(action + "\r\n");

	body.append("--" + boundary + "\r\n");
	body.append("Content-Disposition: form-data; name=\"file\"; filename=\"" + filename + "\"\r\n");
	body.append("Content-Type: image/png\r\n\r\n");

	body.append(file.readAll());
	body.append("\r\n");

	body.append("--" + boundary + "--\r\n");

	//QNetworkAccessManager *networkAccessManager = new QNetworkAccessManager(this);
	QNetworkRequest request(QUrl(_url.toUtf8().data()));
	request.setRawHeader(QString("Content-Type").toAscii(), QString("multipart/form-data; boundary=" + boundary).toAscii());
	request.setRawHeader(QString("Content-Length").toAscii(), QString::number(body.length()).toAscii());





	QEventLoop eventLoop;
 
    // "quit()" the event-loop, when the network request "finished()"
    QNetworkAccessManager mgr;
    QObject::connect(&mgr, SIGNAL(finished(QNetworkReply*)), &eventLoop, SLOT(quit()));
	QNetworkRequest req(QUrl(_url.toUtf8().data()));
    QNetworkReply *re = mgr.post(req, body);
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
		_error = re->errorString();
		delete re;
		return false;
    }




	//QNetworkReply *reply = networkAccessManager->post(request,dataToSend); // perform POST request
}

QString HTTPClient::reply() {
	return _reply;
}

QString HTTPClient::error() {
	return _error;
}
