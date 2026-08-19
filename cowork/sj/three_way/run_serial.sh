#!/usr/bin/env bash
# GPU 작업 직렬화.
#
# Git Bash 에는 flock 이 없다 (예전 판은 조용히 실패해서 락이 전혀 안 걸렸고,
# 그 탓에 세 작업이 GPU 를 나눠 물어 fit 하나가 42초에서 990초가 됐다).
# mkdir 은 원자적이므로 그걸로 락을 만든다.
#
# 사용
#   ./run_serial.sh "bash /tmp/job.sh" > out.log 2>&1 &
LOCK="/tmp/three_way_gpu.lockdir"
while ! mkdir "$LOCK" 2>/dev/null; do
  # 소유 프로세스가 죽었으면 30분 뒤 자동 해제
  if [ -f "$LOCK/pid" ] && ! kill -0 "$(cat "$LOCK/pid" 2>/dev/null)" 2>/dev/null; then
    rm -rf "$LOCK"
    continue
  fi
  sleep 15
done
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT INT TERM
cd "C:/Users/isj67/Desktop/LA9elephantmiracle/cowork/sj/three_way"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
eval "$@"
