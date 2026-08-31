#!/bin/bash
# yn A3 1단계 ablation — GPU 하나라 직렬. 각 arm 3시드 x 2폴드.
cd /c/Users/isj67/Desktop/LA9elephantmiracle/cowork/sj/sj_final/src
LOG=/c/Users/isj67/AppData/Local/Temp/claude/C--Users-isj67-Desktop-LA9elephantmiracle/cb42b045-be93-458c-8dba-b809f055cfa9/scratchpad/yna3
for spec in "YNA3_cb_base:id_freq:cb" "YNA3_cb_a3:id_freq+futures3:cb" "YNA3_mlp_base:id_freq:mlp" "YNA3_mlp_a3:id_freq+futures3:mlp"; do
  NAME="${spec%%:*}"; REST="${spec#*:}"; AT="${REST%%:*}"; MD="${REST##*:}"
  echo "=== $(date +%H:%M:%S) START $NAME (atoms=$AT models=$MD)"
  python run_arm.py --name "$NAME" --atoms "$AT" --models "$MD" --seeds 3 --folds 2024,2022 > "$LOG/$NAME.log" 2>&1
  echo "=== $(date +%H:%M:%S) DONE $NAME rc=$?"
  tail -6 "$LOG/$NAME.log"
done
echo "=== ALL DONE $(date +%H:%M:%S)"
