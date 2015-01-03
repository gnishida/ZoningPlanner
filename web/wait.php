<html>
<head>
<title>Participatory Zone Planning</title>
<link rel="stylesheet" type="text/css" href="style.css">
<script src="jquery.min.js" type="text/javascript"></script>
<script src="jquery.periodicalupdater.js" type="text/javascript"></script>

<script type="text/javascript">
$(document).ready(function(){
    $.PeriodicalUpdater({
        url: 'http://gnishida.site90.com/?cmd=check_done',
        minTimeout: 6000,    // &#36865;&#20449;&#12452;&#12531;&#12479;&#12540;&#12496;&#12523;(&#12511;&#12522;&#31186;)
//      method               // 'post'/'get'&#65306;&#12522;&#12463;&#12456;&#12473;&#12488;&#12513;&#12477;&#12483;&#12489;
//      sendData             // &#36865;&#20449;&#12487;&#12540;&#12479;
//      maxTimeout           // &#26368;&#38263;&#12398;&#12522;&#12463;&#12456;&#12473;&#12488;&#38291;&#38548;(&#12511;&#12522;&#31186;)
//      multiplier           // &#12522;&#12463;&#12456;&#12473;&#12488;&#38291;&#38548;&#12398;&#22793;&#26356;(2&#12395;&#35373;&#23450;&#12398;&#22580;&#21512;&#12289;&#12524;&#12473;&#12509;&#12531;&#12473;&#20869;&#23481;&#12395;&#22793;&#26356;&#12364;&#12394;&#12356;&#12392;&#12365;&#12399;&#12289;&#12522;&#12463;&#12456;&#12473;&#12488;&#38291;&#38548;&#12364;2&#20493;&#12395;&#12394;&#12387;&#12390;&#12356;&#12367;)
//      type                 // xml&#12289;json&#12289;script&#12418;&#12375;&#12367;&#12399;html (jquery.get&#12420;jquery.post&#12398;dataType)
    },
    function(data){
        var myHtml = 'The data: ' + data + '';
        $('#test').prepend(myHtml);
    });
})
</script>

</head>
<body>

<div id="task">
<h1>Please wait for a moment...</h1>
<form>
<p>Thank you for answering questions. Please wait for a moment while other designers are finishing the tasks.</p>
</form>
</div>
<div id=test>
</div>
</body>
</html>
