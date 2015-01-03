<?
$error_msg = "";

if (!empty($_REQUEST["email"])) {
	$_SESSION["email"] = $_REQUEST["email"];
	
	include("db_connect.php");
	
	$sql = "SELECT * FROM users WHERE email = '" . $_REQUEST["email"] . "'";
	$result = mysql_query($sql);
	if (!$result) {
    	die('DB query error: ' . mysql_error());
	}
	$row = mysql_fetch_assoc($result);
	if ($row) {
		$user_id = $row["user_id"];
		
		$_SESSION["user_id"] = $row["user_id"];
		$_SESSION["email"] = $row["email"];
		
		// get the current round
		$sql = "SELECT * FROM round";
		$result = mysql_query($sql);
		$row = mysql_fetch_assoc($result);
		$round = 0;
		if ($row) {
			$round = $row["round"];
		}
		
		$sql = "DELETE FROM choices WHERE user_id = " . $user_id . " AND round > " . $round;
		$result = mysql_query($sql);
		if (!$result) {
    		die('DB query error: ' . mysql_error());
		}
		
		$sql = "SELECT IFNULL(MAX(step), 0) max_step FROM choices WHERE user_id = " . $user_id . " AND round = " . $round;
		$result = mysql_query($sql);
		if (!$result) {
    		die('DB query error: ' . mysql_error());
		}
		$row = mysql_fetch_assoc($result);
		$step = 0;
		if ($row) {
			$step = $row["max_step"];
		}
		$step = $step + 1;
		
		if ($step <= 3) {
			header("Location: http://gnishida.site90.com/?cmd=design&round=" . $round . "&step=" . $step);
			exit;
		} else {
			header("Location: http://gnishida.site90.com/?cmd=wait&round=" . $round);
			exit;
		}
	} else {
		$error_msg = "The email you entered is incorrect.";
	}
}
?>

<html>
<head>
<title>Participatory Zone Planning</title>
<link rel="stylesheet" type="text/css" href="style.css">
</head>
<body>

<div id="login">
<h1>Login</h1>
<form action="http://gnishida.site90.com/?cmd=login" method="GET">
<input type="email" name="email" placeholder="Email" /><br/>
<p class="error"><?= $error_msg ?></p>
<input type="submit" value="Login"/><br/>
</form>
</div>
</body>
</html>
