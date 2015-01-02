<?
session_start();

$cmd = htmlspecialchars($_REQUEST["cmd"]);

if ($cmd == '') $cmd = 'login';

include $cmd . '.php';
?>
