<?
require("util.php");
connect_db();

$step = $_REQUEST["step"];
$option1 = $_REQUEST["option1"];
$option2 = $_REQUEST["option2"];

// get the next round
$sql = "SELECT * FROM round";
$result = mysql_query($sql);
$row = mysql_fetch_assoc($result);
$round = 0;
if ($row) {
	$round = $row["round"];
}
$round = $round + 1;

$sql = "INSERT INTO tasks(round, step, option1, option2) VALUES(" . $round . ", " . $step . ", \"" . $option1 . "\", \"" . $option2 . "\")";
$result = mysql_query($sql);
if (!$result) {
	print("DB query error: " . mysql_error());
} else {
	print("OK");
}
?>
