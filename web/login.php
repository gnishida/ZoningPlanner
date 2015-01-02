<?
if (!empty($_REQUEST["email"])) {
	$_SESSION["email"] = $_REQUEST["email"];
	
	$mysql_host = "mysql4.000webhost.com";
	$mysql_database = "a4249468_zp";
	$mysql_user = "a4249468_gen";
	$mysql_password = "karen99";
	
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
	
	header("Location: http://gnishida.site90.com/?cmd=design&step=1&question=1");
	exit;
}
?>

<html>
<head>
<title>Participatory Zone Planning</title>
</head>
<body>
<form action="http://gnishida.site90.com/?cmd=login" method="GET">
Email: <input type="text" name="email" /><br/>
<input type="submit" value="Login"/><br/>
</form>
</body>
</html>