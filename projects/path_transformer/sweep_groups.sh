#!/usr/bin/env bash
# Rung 2 (group tracking) stage with the long training schedule and a lean
# probe grid. Resumable; results copied into projects/group_tracking/results/runs.
set -u
cd /home/culpritgene/GeoMechInterp
PY=.venv/bin/python; GT=projects/group_tracking; CK=/var/tmp/geomech_ckpt; L=$CK/sweep_logs; G=$GT/results/runs
mkdir -p $G $L /var/tmp/geomech_data/groups
LANES=${LANES:-3}
log() { echo "=== $* $(date '+%H:%M:%S')" | tee -a $L/_sweep.log; }
group_job() {  # group d [tag]
  local g=$1 d=$2 tag=${3:-_long} run=group_${1}_L4_d$2${3:-_long}
  [ -f /var/tmp/geomech_data/groups/${g}_0.npz ] || $PY $GT/group_data.py --group $g --seed 0 --n_train 200000 >> $L/_gen.log 2>&1
  if pgrep -f "train_group.py --group $g --n_layer 4 --d_model $d " >/dev/null; then echo "skip $run: training already running" | tee -a $L/_sweep.log; return; fi
  if [ ! -f $CK/$run/summary.json ]; then
    $PY $GT/train_group.py --group $g --n_layer 4 --d_model $d --n_head $(( d >= 64 ? 8 : 4 )) --steps 120000 --min_steps 60000 --patience 40 --batch 512 --lr 5e-4 --tag $tag > $L/train_$run.log 2>&1
    echo "train done $run $(grep '^TEST' $L/train_$run.log | cut -c1-300)" | tee -a $L/_sweep.log
  fi
  cp $CK/$run/summary.json $G/${run}_summary.json 2>/dev/null
  if pgrep -f "probe_group.py --run $run " >/dev/null; then echo "skip $run: probe already running" | tee -a $L/_sweep.log; return; fi
  if [ ! -f $CK/$run/probes.json ]; then
    $PY $GT/probe_group.py --run $run --layers 0 2 4 --targets irrep eps harm > $L/probe_$run.log 2>&1
    echo "probe done $run" | tee -a $L/_sweep.log
  fi
  cp $CK/$run/probes.json $G/${run}_probes.json 2>/dev/null
  if ! ls $CK/null_control/null_${g}_d${d}*.json >/dev/null 2>&1; then
    $PY $GT/null_control.py --group $g --d $d --freqs 1 > $L/null_${g}_d$d.log 2>&1
    cp $CK/null_control/null_${g}_d${d}*.json $G/ 2>/dev/null
    echo "null done ${g} d$d" | tee -a $L/_sweep.log
  fi
}
J=$L/_group_jobs2.txt; : > $J
for d in 64 256 32; do echo "group_job D36 $d" >> $J; done
for g in T36x12 Z36; do for d in 32 64 256; do echo "group_job $g $d" >> $J; done; done
for d in 32 64; do echo "group_job Z360 $d" >> $J; done
lane() { local id=$1 i=0; while read -r line; do i=$((i+1)); [ $(( (i-1) % LANES )) -eq $id ] && eval "$line"; done < $J; }
log "groups2 start: $(wc -l < $J) jobs, $LANES lanes"; for ((id=0; id<LANES; id++)); do lane $id & done; wait; log "groups2 done"
