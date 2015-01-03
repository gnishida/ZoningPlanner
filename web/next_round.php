<?
require("util.php");
connect_db();

// update the round
$sql = "UPDATE config SET current_round = current_round + 1";
$result = mysql_query($sql);
if (!$result) {
	print("DB update error: " . mysql_error());
} else {
	print("OK");
}
?>
