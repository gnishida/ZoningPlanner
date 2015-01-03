<?
include("db_connect.php");

// update the round
$sql = "UPDATE round SET round = round + 1";
$result = mysql_query($sql);
if (!$result) {
	print("DB update error: " . mysql_error());
} else {
	print("OK");
}
?>
