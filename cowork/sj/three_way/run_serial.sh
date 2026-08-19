#!/usr/bin/env bash
# GPU 작업 직렬화.
#
# nvidia-smi 메모리로 "비었나" 판단하면 fit 사이 메모리 하락에 속아 두 작업이
# 겹친다. 실제로 세 작업이 동시에 GPU 를 물어 fit 하나가 42초에서 990초로
# 늘어난 적이 있다. 락 파일로 확실히 한 번에 하나만 돌게 한다.
#
# 사용
#   ./run_serial.sh "python src/way_weights.py --target reverse ..." > out.log 2>&1 &
set -o pipefail
LOCK="/tmp/three_way_gpu.lock"
exec 9>"$LOCK"
flock 9
cd "C:/Users/isj67/Desktop/LA9elephantmiracle/cowork/sj/three_way"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
eval "$@"
