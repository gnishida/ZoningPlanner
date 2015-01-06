<?
require("util.php");

$db = connect_db();
list($current_round, $max_round, $max_step) = get_config();

$email = $_SESSION["email"];

$round = $_REQUEST["round"];
if (empty($_REQUEST["round"])) {
	$round = 1;
}



if ($round >= $max_round) {
	header("Location: http://gnishida.site90.com/?cmd=complete");
	exit;
}
?>

<html>
<head>
<title>Participatory Zone Planning</title>
<link rel="stylesheet" type="text/css" href="style.css">
<script src="jquery.min.js" type="text/javascript"></script>
<script src="smartupdater.4.0.js" type="text/javascript"></script>

<script type="text/javascript">
$(document).ready(function() {

$("#task").smartupdater({
	url : 'check_done.php',
	minTimeout: 10000 // 10 seconds
	}, function (data) {
		if (data == <?= $round + 1 ?>) {
			window.location.href = "http://gnishida.site90.com/?cmd=design&round=" + <?= $round + 1?> + "&step=1";
		}
	}
);

});
</script>

</head>
<body>

<div id="task">
<p class="right"><?=$email?></p>

<h1>Please wait for a moment...</h1>
<form>
<p>Thank you for answering questions. Please wait for a moment while other designers are finishing their tasks.</p>
<img src="loading.gif"/>
</form>
</div>
</body>
</html>
