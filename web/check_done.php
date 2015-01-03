<?
$link = mysql_connect("mysql4.000webhost.com", "a4249468_gen", "karen99");
if (!$link) {
	die('DB connection error: ' . mysql_error());
}

$db = mysql_select_db("a4249468_zp", $link);
if (!$db) {
	die('DB select error: ' . mysql_error());
}

$min_round = 99;

$sql = "SELECT u.user_id, u.email, IFNULL(MAX(c.round), 0) max_round FROM users u LEFT OUTER JOIN choices c ON u.user_id = c.user_id WHERE c.step = 3 GROUP BY u.user_id";
$result = mysql_query($sql);
while ($row = mysql_fetch_assoc($result)) {
	if ($row["max_round"] < $min_round) {
		$min_round = $row["max_round"];
	}
}

print($min_round);
?>
