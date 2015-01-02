<html>
<head>
<title>Participatory Zone Planning</title>
<link rel="stylesheet" type="text/css" href="style.css">
<meta http-equiv="refresh" content="5;URL=http://gnishida.site90.com/?cmd=admin">
</head>
<body>

<?
$link = mysql_connect("mysql4.000webhost.com", "a4249468_gen", "karen99");
if (!$link) {
	die('DB connection error: ' . mysql_error());
}

$db = mysql_select_db("a4249468_zp", $link);
if (!$db) {
	die('DB select error: ' . mysql_error());
}

$sql = "SELECT u.user_id, u.email, IFNULL(MAX(c.round), 0) max_round, IFNULL(MAX(c.step), 0) max_step FROM users u LEFT OUTER JOIN choices c ON u.user_id = c.user_id GROUP BY u.user_id ORDER BY u.user_id";
$result = mysql_query($sql);
if (!$result) {
	die('DB query error: ' . mysql_error());
}
?>
<table>
<tbody>
<tr><th>user_id</th><th>email</th><th>round</th><th>step</th></tr>
<?
while ($row = mysql_fetch_assoc($result)) {
?>
<tr><td><?= $row["user_id"] ?></td><td><?= $row["email"] ?></td><td><?= $row["max_round"]?></td><td><?= $row["max_step"]?></td></tr>

<?
}
?>

</tbody>
</table>

</body>
</html>