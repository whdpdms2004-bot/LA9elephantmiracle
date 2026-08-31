#!/bin/bash
# 1단계 확인 — 8시드 paired. 3시드 여유폭(관문1 +0.6, 2023R +0.3)이 시드잡음 안이라 다시 잰다.
cd /c/Users/isj67/Desktop/LA9elephantmiracle/cowork/sj/sj_final/src
LOG=/c/Users/isj67/AppData/Local/Temp/claude/C--Users-isj67-Desktop-LA9elephantmiracle/cb42b045-be93-458c-8dba-b809f055cfa9/scratchpad/yna3
for spec in "S8_cb_base:id_freq:cb" "S8_cb_a3:id_freq+futures3:cb" "S8_mlp_base:id_freq:mlp" "S8_mlp_a3:id_freq+futures3:mlp"; do
  NAME="${spec%%:*}"; REST="${spec#*:}"; AT="${REST%%:*}"; MD="${REST##*:}"
  echo "=== $(date +%H:%M:%S) START $NAME"
  python run_arm.py --name "$NAME" --atoms "$AT" --models "$MD" --seeds 8 --folds 2024,2022,2023 > "$LOG/$NAME.log" 2>&1
  echo "=== $(date +%H:%M:%S) DONE rc=$?"
  grep -E "^  20[0-9][0-9]|    all " "$LOG/$NAME.log" | head -6
done
echo "=== ALL DONE $(date +%H:%M:%S)"
