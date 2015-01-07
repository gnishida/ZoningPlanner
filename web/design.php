<?
$user_id = $_SESSION["user_id"];
$email = $_SESSION["email"];
$round = $_REQUEST["round"];
$step = $_REQUEST["step"];

if (empty($_SESSION["user_id"])) {
	header("Location: http://gnishida.site90.com/?cmd=login");
	exit;
}

require("util.php");

connect_db();
list($current_round, $max_round, $max_step) = get_config();

if (!empty($_REQUEST["choice"])) {
	$choice = $_REQUEST["choice"];
	$sql = "INSERT INTO choices(user_id, round, step, choice) VALUES(" . $user_id . ", " . $round . ", " . $step . ", " . $choice . ")";
	$result = mysql_query($sql);
	if (!$result) {
		die('DB insert error: ' . mysql_error());
	}
	
	$step = $step + 1;
	if ($step > $max_step) {
		if ($round >= $max_round) {
			header("Location: http://gnishida.site90.com/?cmd=complete");
			exit;
		} else {
			header("Location: http://gnishida.site90.com/?cmd=wait&round=" . $round);
			exit;
		}
	}
}


$sql = "SELECT * FROM tasks WHERE round = " . $round . " AND step = " . $step;
$result = mysql_query($sql);
if (!$result) {
	die('DB query error: ' . mysql_error());
}

$row = mysql_fetch_assoc($result);
$feature1 = explode(",", $row["option1"]);
$feature2 = explode(",", $row["option2"]);
?>

<html>
<head>
<title>Participatory Zone Planning</title>
<link rel="stylesheet" type="text/css" href="style.css">
</head>
<body>
<div id="task">
<p class="right"><?=$email?></p>
<h1>Round <?= $round ?> - Question <?= $step ?> / <?= $max_step ?></h1>

<form action="http://gnishida.site90.com/" method="GET">
<input type="hidden" name="cmd" value="design"/>
<input type="hidden" name="round" value="<?= $round?>"/>
<input type="hidden" name="step" value="<?= $step?>"/>

<table border="1">
<tr><th></th><th>Option 1</th><th>Option 2</th><tr>
<tr><td>To the nearest store</td><td><?= $feature1[0] ?> [m]</td><td><?= $feature2[0] ?> [m]</td></tr>
<tr><td>To the nearest school</td><td><?= $feature1[1] ?> [m]</td><td><?= $feature2[1] ?> [m]</td></tr>
<tr><td>To the nearest restaurant</td><td><?= $feature1[2] ?> [m]</td><td><?= $feature2[2] ?> [m]</td></tr>
<tr><td>To the nearest park</td><td><?= $feature1[3] ?> [m]</td><td><?= $feature2[3] ?> [m]</td></tr>
<tr><td>To the nearest amusement facility</td><td><?= $feature1[4] ?> [m]</td><td><?= $feature2[4] ?> [m]</td></tr>
<tr><td>To the nearest library</td><td><?= $feature1[5] ?> [m]</td><td><?= $feature2[5] ?> [m]</td></tr>
<tr><td>To the nearest factory</td><td><?= $feature1[6] ?> [m]</td><td><?= $feature2[6] ?> [m]</td></tr>
</table>

<p>Which option do you prefer?
<input type="radio" name="choice" value="1" id="choice1"/><label for="choice1">Option 1</label>
<input type="radio" name="choice" value="2" id="choice2"/><label for="choice2">Option 2</label></p>
<input type="submit" value="Submit"/>
</form>

</div>

<?
if (file_exists("images/" . $user_id . "_" . ($round - 1) . ".png")) {
?>
<div id="picture">
<h2>The best place for you to live in according to your answers</h2>
<p class="small">Red - Residential / Blue - Commercial / Gray - Industorial / Green - Park / Cyan - School and Library / Yellow - Amusement facilities</p>
<img width="480" src="images/<?= $user_id ?>_<?= $round-1 ?>.png" />
</div>
<?
}
?>

</body>
</html>