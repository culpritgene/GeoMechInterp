#!/usr/bin/env bash
# Lean probe stage for the manifold sweep (middle layer only, pos/goal/dir),
# then the winding stage. Resumable: skips models that already have probes_v3.json.
set -u
cd /home/culpritgene/GeoMechInterp
PY=.venv/bin/python; PT=projects/path_transformer; CK=/var/tmp/geomech_ckpt; L=$CK/sweep_logs; RUNS=$PT/results/runs
LANES=${LANES:-4}
log() { echo "=== $* $(date '+%H:%M:%S')" | tee -a $L/_sweep.log; }
probe_job() {
  local run=$1
  if [ -f $CK/$run/probes_v3.json ] || [ ! -f $CK/$run/best.pt ]; then cp $CK/$run/probes_v3.json $RUNS/${run}_probes_v3.json 2>/dev/null; return; fi
  case $run in *L3_d64) LAY="2";; *) LAY="4";; esac
  $PY $PT/probe.py --run $run --layers $LAY --targets pos goal dir --n_seq 3000 --n_train 40000 --steps 3000 --out $CK/$run/probes_v3.json > $L/probe_$run.log 2>&1
  cp $CK/$run/probes_v3.json $RUNS/${run}_probes_v3.json 2>/dev/null
  echo "probe done $run" | tee -a $L/_sweep.log
}
J=$L/_probe_jobs_lean.txt; : > $J
for d in $CK/{ring,chain,single_ring,keyring_1}*_flat_L*_d*; do [ -d $d ] && echo "probe_job $(basename $d)" >> $J; done
sort -u $J -o $J
lane() { local id=$1 i=0; while read -r line; do i=$((i+1)); [ $(( (i-1) % LANES )) -eq $id ] && eval "$line"; done < $J; }
log "lean probes start: $(wc -l < $J) jobs, $LANES lanes"; for ((id=0; id<LANES; id++)); do lane $id & done; wait
log "probes done"
STAGES=winding bash $PT/sweep.sh
