<?
include("db_connect.php");

$sql = "SELECT * FROM round";
$result = mysql_query($sql);
$row = mysql_fetch_assoc($result);
if ($row) {
	print($row["round"]);
} else {
	print(0);
}

?>
