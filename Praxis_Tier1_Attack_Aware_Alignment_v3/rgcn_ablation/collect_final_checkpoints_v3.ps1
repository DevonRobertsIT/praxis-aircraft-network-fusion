# v2: gathers checkpoints, JSON metadata, result CSVs AND the preprocessing folder (event/flow preprocessors) for the companion package.
$COL = "$PX\final_checkpoints"; if (Test-Path $COL) { Remove-Item -Recurse -Force $COL }
function Grab($s, $d) {
  if (-not (Test-Path $s)) { return }
  New-Item -ItemType Directory -Force "$d\models", "$d\preprocessing" | Out-Null
  Copy-Item "$s\models\*" "$d\models" -ErrorAction SilentlyContinue
  Copy-Item "$s\preprocessing\*" "$d\preprocessing" -ErrorAction SilentlyContinue
  Copy-Item "$s\*.json" $d -ErrorAction SilentlyContinue; Copy-Item "$s\*_results_v2.csv" $d -ErrorAction SilentlyContinue
}
foreach ($SEED in 42,43,44,45,46) {
  Grab "$ROOT\data\processed\point_event_v2\transformer_v3_seed$SEED" "$COL\regime_seed$SEED\transformer_v2"
  Grab "$ROOT\data\processed\point_event_v2\rgcn_trained_v3_seed$SEED" "$COL\regime_seed$SEED\rgcn_trained_v3" }
Get-ChildItem "$CELLS" -Directory | ForEach-Object {
  foreach ($sub in "transformer_v2","rgcn_trained_v3","rgcn_trained_v32") { Grab "$($_.FullName)\data\processed\point_event_v2\$sub" "$COL\$($_.Name)\$sub" } }
Compress-Archive -Path "$COL\*" -DestinationPath "$PX\final_checkpoints.zip" -Force
"pt files: $((Get-ChildItem $COL -Recurse -Filter *.pt).Count); preprocessors: $((Get-ChildItem $COL -Recurse -Filter *.joblib).Count)"
