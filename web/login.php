<?
if (!empty($_REQUEST["email"])) {
	$_SESSION["email"] = $_REQUEST["email"];
	
	$link = mysql_connect("mysql4.000webhost.com", "a4249468_gen", "karen99");
	if (!$link) {
    	die('connection error: ' . mysql_error());
	}
	
	$db = mysql_select_db("a4249468_zp", $link);
	if (!$db) {
    	die('db select error: ' . mysql_error());
	}
	
	$sql = "INSERT INTO users(email) VALUES('" . $_REQUEST["email"] . "')";
	$result = mysql_query($sql);
	if (!$result) {
    	die('insert error: ' . mysql_error());
	}
	
	$sql = "SELECT * FROM users WHERE email = '" . $_REQUEST["email"] . "'";
	$result = mysql_query($sql);
	if (!$result) {
    	die('insert error: ' . mysql_error());
	}
	$row = mysql_fetch_assoc($result);
	$_SESSION["user_id"] = $row["user_id"];
	
	header("Location: http://gnishida.site90.com/?cmd=design&round=1&step=1");
	exit;
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
<input type="submit" value="Login"/><br/>
</form>
</div>
</body>
</html>
