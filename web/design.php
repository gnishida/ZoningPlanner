<html>
<head>
<title>Participatory Zone Planning</title>
</head>
<body>

<?
$step = $_REQUEST["step"];
$question = $_REQUEST["question"];
?>

<h2>Step <?= $step ?></h2>
<h3>Question <?= $question ?></h3>

<form action="http://gnishida.site90.com/" method="GET">
<input type="hidden" name="cmd" value="design"/>
<input type="hidden" name="step" value="<?= $step?>"/>
<input type="hidden" name="question" value="<?= $question + 1?>"/>
<input type="submit" value="submit"/>
</form>

</body>
</html>