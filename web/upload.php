<?
if (isset($_FILES["file"])) {
	$filename = $_FILES["file"]["name"];
	$tmp_path = $_FILES["file"]["tmp_name"];
	
	if (!move_uploaded_file($tmp_path, "images/" . $filename)) {
		die("Error: cannot save the file.");
	}
	
	print("OK");
} else {
	print("Error: invalid request.");
}
?>