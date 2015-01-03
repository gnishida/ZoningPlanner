<?
include("db_connect.php");

// get the current round
$sql = "SELECT * FROM round";
$result = mysql_query($sql);
$row = mysql_fetch_assoc($result);
$round = 0;
if ($row) {
	$round = $row["round"];
}

// get the choices
$user_ids = array();
$sql = "SELECT user_id FROM choices WHERE round = " . $round . " GROUP BY user_id";
$result = mysql_query($sql);
while ($row = mysql_fetch_assoc($result)) {
	$user_ids[] = $row["user_id"];
}

$data = array();
foreach ($user_ids as $user_id) {
	$sql = "SELECT * FROM choices WHERE round = " . $round . " AND user_id = " . $user_id . " ORDER BY step";
	$result = mysql_query($sql);
	$record = array("user_id" => $user_id);
	while ($row = mysql_fetch_assoc($result)) {
		$record[$row["step"]] = $row["choice"];
	}
	$data[] = $record;
}

header("Content-type: application/json");
echo json_encode($data);
?>
