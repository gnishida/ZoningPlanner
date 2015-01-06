<?
require("util.php");
connect_db();

// Configure the system parameters.
$sql = "SELECT * FROM config";
$result = mysql_query($sql);
$row = mysql_fetch_assoc($result);

print($row["current_round"]);
?>