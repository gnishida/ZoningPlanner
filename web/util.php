<?
function connect_db() {
	$link = mysql_connect("mysql4.000webhost.com", "a4249468_gen", "karen99");
	if (!$link) {
		die('DB connection error: ' . mysql_error());
	}

	$db = mysql_select_db("a4249468_zp", $link);
	if (!$db) {
		die('DB select error: ' . mysql_error());
	}
	
	return $db;
}

function get_config() {
	$sql = "SELECT * FROM config";
	$result = mysql_query($sql);
	if (!$result) {
		die('DB insert error: ' . mysql_error());
	}
	$row = mysql_fetch_assoc($result);
	if ($row) {
		return array($row["current_round"], $row["max_round"], $row["max_step"]);
	} else {
		return array(0, 0, 0);
	}
}
?>
