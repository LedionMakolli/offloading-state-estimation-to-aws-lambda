# Capture the Pico's USB-CDC output to a file and to the console.
#
#   powershell -File tools\read_serial.ps1 -Port COM10 -Seconds 240 -Out ..\docs\e4b_results.txt
#
# Stops early on the [done] or [fail] marker so a finished run does not sit
# waiting out the whole timeout.
param(
    [string]$Port = "COM10",
    [int]$Seconds = 240,
    [string]$Out = "",
    [string]$StopOn = "\[done\]|\[fail\]"
)

$sp = New-Object System.IO.Ports.SerialPort $Port, 115200, 'None', 8, 'One'
$sp.ReadTimeout = 500
$sp.DtrEnable = $true          # the SDK waits for DTR before it prints
$sp.NewLine = "`n"

try { $sp.Open() } catch { Write-Output "cannot open $Port : $_"; exit 1 }

$sb = New-Object System.Text.StringBuilder
$deadline = (Get-Date).AddSeconds($Seconds)
$pending = ""

while ((Get-Date) -lt $deadline) {
    try { $chunk = $sp.ReadExisting() } catch { $chunk = "" }
    if ($chunk.Length -gt 0) {
        $pending += $chunk
        # Emit whole lines only, so a line split across two reads is not torn.
        while ($pending.Contains("`n")) {
            $i = $pending.IndexOf("`n")
            $line = $pending.Substring(0, $i).TrimEnd("`r")
            $pending = $pending.Substring($i + 1)
            Write-Output $line
            [void]$sb.AppendLine($line)
        }
        if ($sb.ToString() -match $StopOn) { break }
    } else {
        Start-Sleep -Milliseconds 100
    }
}

if ($pending.Length -gt 0) { Write-Output $pending; [void]$sb.Append($pending) }
$sp.Close()

if ($Out -ne "") {
    $sb.ToString() | Out-File -FilePath $Out -Encoding utf8
    Write-Output ""
    Write-Output "saved to $Out"
}
