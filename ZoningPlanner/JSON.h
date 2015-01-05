#pragma once

#include <vector>
#include <QString>

class JSON {
private:
	JSON() {}

public:
	static std::vector<std::pair<QString, QString> > parse(const QString& json_data, const QString& path, const QString& node1, const QString& node2);
};

