#!/bin/bash
# 관문 4 (val2023 R 비하락) 을 재려면 2023 폴드가 필요하다.
cd /c/Users/isj67/Desktop/LA9elephantmiracle/cowork/sj/sj_final/src
LOG=/c/Users/isj67/AppData/Local/Temp/claude/C--Users-isj67-Desktop-LA9elephantmiracle/cb42b045-be93-458c-8dba-b809f055cfa9/scratchpad/yna3
for spec in "YNA3_cb_base:id_freq:cb" "YNA3_cb_a3:id_freq+futures3:cb" "YNA3_mlp_base:id_freq:mlp" "YNA3_mlp_a3:id_freq+futures3:mlp"; do
  NAME="${spec%%:*}"; REST="${spec#*:}"; AT="${REST%%:*}"; MD="${REST##*:}"
  echo "=== $(date +%H:%M:%S) START ${NAME}_2023"
  python run_arm.py --name "${NAME}_f23" --atoms "$AT" --models "$MD" --seeds 3 --folds 2023 > "$LOG/${NAME}_f23.log" 2>&1
  echo "=== $(date +%H:%M:%S) DONE rc=$?"
  grep -E "  2023|all " "$LOG/${NAME}_f23.log" | head -3
done
echo "=== ALL DONE $(date +%H:%M:%S)"
