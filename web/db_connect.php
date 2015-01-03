<?
$link = mysql_connect("mysql4.000webhost.com", "a4249468_gen", "karen99");
if (!$link) {
	die('DB connection error: ' . mysql_error());
}

$db = mysql_select_db("a4249468_zp", $link);
if (!$db) {
	die('DB select error: ' . mysql_error());
}
?>
