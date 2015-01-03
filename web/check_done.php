<?
require("util.php");

connect_db();

$sql = "SELECT current_round FROM config";
$result = mysql_query($sql);
$row = mysql_fetch_assoc($result);
if ($row) {
	print($row["current_round"]);
} else {
	print(0);
}

?>
