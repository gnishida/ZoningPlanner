<?
require("util.php");
connect_db();

// Configure the system parameters.
$current_round = mysql_real_escape_string($_REQUEST["current_round"]);
$max_round = mysql_real_escape_string($_REQUEST["max_round"]);
$max_step = mysql_real_escape_string($_REQUEST["max_step"]);

$sql = "TRUNCATE config";
$result = mysql_query($sql);
if (!$result) {
    die('DB truncate error: ' . mysql_error());
}

$sql = "INSERT INTO config(current_round, max_round, max_step) VALUES(" . $current_round . ", " . $max_round . ", " . $max_step . ")";
$result = mysql_query($sql);
if (!$result) {
    die('DB insert error: ' . mysql_error());
}


// delete the old records
$sql = "TRUNCATE choices";
$result = mysql_query($sql);
if (!$result) {
    die('DB truncate error: ' . mysql_error());
}
$sql = "TRUNCATE tasks";
$result = mysql_query($sql);
if (!$result) {
    die('DB truncate error: ' . mysql_error());
}

// delete the old pictures
if ($dir = opendir("images/")) {
	while (($file = readdir($dir)) !== false) {
		if ($file != "." && $file != "..") {
			unlink("images/" . $file);
		}
	}
	closedir($dir);
}

print("OK");
?>