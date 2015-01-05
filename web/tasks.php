<?
require("util.php");
connect_db();
list($round, $max_round, $max_step) = get_config();

// get the tasks
$user_ids = array();
$sql = "SELECT * FROM tasks WHERE round = " . $round . " GROUP BY step";
$result = mysql_query($sql);
$tasks = array();
while ($row = mysql_fetch_assoc($result)) {
	$record["option1"] = $row["option1"];
	$record["option2"] = $row["option2"];
	$tasks[] = $record;
}

$data["tasks"] = $tasks;

header("Content-type: application/json");
echo json_encode($data);
?>
