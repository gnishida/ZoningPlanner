<?
require("util.php");
connect_db();

$step = $_REQUEST["step"];
$option1 = $_REQUEST["option1"];
$option2 = $_REQUEST["option2"];

// get the config
list($current_round, $max_round, $max_step) = get_config();
$round = $current_round + 1;

$sql = "SELECT * FROM tasks WHERE round = " . $round . " AND step = " . $step;
$result = mysql_query($sql);
$num = mysql_num_rows($result);
if ($num == 0) {
	$sql = "INSERT INTO tasks(round, step, option1, option2) VALUES(" . $round . ", " . $step . ", '" . $option1 . "', '" . $option2 . "')";
	$result = mysql_query($sql);
	if (!$result) {
		die("DB query error: " . mysql_error());
	}
} else {
	$sql = "UPDATE tasks SET option1 = '" . $option1 . "', option2 = '" . $option2 . "' WHERE round = " . $round . " AND step = " . $step;
	$result = mysql_query($sql);
	if (!$result) {
		die("DB query error: " . mysql_error());
	}
}


print("OK");
?>
