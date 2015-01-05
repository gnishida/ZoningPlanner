#include "HTTPClient.h"
#include <QEventLoop>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QHttp>
#include <QFile>

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

void HTTPClient::uploadFile(const QString& hostname, const QString& action, const QString& filename) {
	QHttp* http = new QHttp(); // http declared as a member of MainWindow class

    QString boundary = "---------------------------723690991551375881941828858";

    // action
    QByteArray data(QString("--" + boundary + "\r\n").toAscii());
    data += "Content-Disposition: form-data; name=\"action\"\r\n\r\n";
    data += "file_upload\r\n";

    // file
    data += QString("--" + boundary + "\r\n").toAscii();
    data += "Content-Disposition: form-data; name=\"sfile\"; filename=\"" + filename + "\"\r\n";
    data += "Content-Type: image/png\r\n\r\n";

    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly))
        return;

    data += file.readAll();
    data += "\r\n";

    // password
    data += QString("--" + boundary + "\r\n").toAscii();
    data += "Content-Disposition: form-data; name=\"password\"\r\n\r\n";
    //data += "password\r\n"; // put password if needed
    data += "\r\n";

    // description
    data += QString("--" + boundary + "\r\n").toAscii();
    data += "Content-Disposition: form-data; name=\"description\"\r\n\r\n";
    data += "\r\n";

    // agree
    data += QString("--" + boundary + "\r\n").toAscii();
    data += "Content-Disposition: form-data; name=\"agree\"\r\n\r\n";
    data += "1\r\n";

    data += QString("--" + boundary + "--\r\n").toAscii();

    QHttpRequestHeader header("POST", action);
    header.setValue("Host", hostname);
    header.setValue("User-Agent", "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.1.9) Gecko/20100401 Ubuntu/9.10 (karmic) Firefox/3.5.9");
    header.setValue("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8");
    header.setValue("Accept-Language", "en-us,en;q=0.5");
    header.setValue("Accept-Encoding", "gzip,deflate");
    header.setValue("Accept-Charset", "ISO-8859-1,utf-8;q=0.7,*;q=0.7");
    header.setValue("Keep-Alive", "300");
    header.setValue("Connection", "keep-alive");
    header.setValue("Referer", "http://" + hostname + "/");

    //multipart/form-data; boundary=---------------------------723690991551375881941828858

    header.setValue("Content-Type", "multipart/form-data; boundary=" + boundary);
    header.setValue("Content-Length", QString::number(data.length()));

    http->setHost(hostname);
    http->request(header, data);

    file.close();
}

QString HTTPClient::reply() {
	return _reply;
}

QString HTTPClient::error() {
	return _error;
}
