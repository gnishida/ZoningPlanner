<?
$round = $_REQUEST["round"];
if (empty($_REQUEST["round"])) {
	$round = 1;
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

$("#example1").smartupdater({
	url : 'check_done.php',
	minTimeout: 5000 // 5 seconds
	}, function (data) {
		if (data < <?= $round ?>) {
			$("#example1").val("wait!! " + data);
		} else {
			window.location.href = "http://gnishida.site90.com/?cmd=design&round=" + <?= $round + 1?> + "&step=1";
		}
	}
);

});
</script>

</head>
<body>

<div id="task">
<h1>Please wait for a moment...</h1>
<form>
<p>Thank you for answering questions. Please wait for a moment while other designers are finishing their tasks.</p>
</form>
</div>
<input id="example1" type="text" value="" />
</body>
</html>
