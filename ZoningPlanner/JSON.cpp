#include "JSON.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <cassert>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>

std::vector<std::pair<QString, QString> > JSON::parse(const QString& json_data, const QString& path, const QString& node1, const QString& node2) {
	std::stringstream ss;

	ss << json_data.toUtf8().data();

	//ss << "{ \"root\": { \"values\": [1, 2, 3, 4, 5 ] } }";
	//ss << "{\"results\": [{ \"user_id\":4, \"choices\": \"1,1,1\" }, { \"user_id\":16, \"choices\": \"2,2,1\" }]}";
 
	boost::property_tree::ptree pt;
	boost::property_tree::read_json(ss, pt);
 
	for (auto child = pt.begin(); child != pt.end(); ++child) {
		std::cout << child->first.data() << std::endl;
		std::cout << child->second.data() << std::endl;
	}
	
	std::vector<std::pair<QString, QString> > ret;
	BOOST_FOREACH(boost::property_tree::ptree::value_type &v, pt.get_child(path.toUtf8().data())) {
		ret.push_back(std::make_pair(QString::fromStdString(v.second.get<std::string>(node1.toUtf8().data())), QString::fromStdString(v.second.get<std::string>(node2.toUtf8().data()))));
	}

	return ret;
}
