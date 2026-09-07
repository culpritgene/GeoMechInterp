#!/usr/bin/env bash
# Resumable manifold sweep: manifolds -> path datasets -> path models -> probes,
# plus the rung 1 (winding) and rung 2 (group tracking) runs.
# Every stage skips outputs that already exist, so the script can be re-run
# after an interruption. Summaries are copied into the repo (on /home) as soon
# as each job finishes, because /var/tmp is wiped on reboot.
#
#   bash projects/path_transformer/sweep.sh            # everything
#   STAGES="data train" bash projects/path_transformer/sweep.sh
set -u
cd /home/culpritgene/GeoMechInterp
PY=.venv/bin/python
PT=projects/path_transformer
GT=projects/group_tracking
CK=/var/tmp/geomech_ckpt
GEN=/var/tmp/geomech_data/generated
MAN=/var/tmp/geomech_data/manifolds
RUNS=$PT/results/runs
GRUNS=$GT/results/runs
L=$CK/sweep_logs
mkdir -p $CK/logs $L $GEN $MAN /var/tmp/geomech_data/groups $RUNS $GRUNS
STAGES=${STAGES:-"data train probes winding groups"}
LANES=${LANES:-5}
log() { echo "=== $* $(date '+%H:%M:%S')" | tee -a $L/_sweep.log; }

# ---------------------------------------------------------------- datasets
MANIFOLDS=(
  "ring_r012   --preset ring --r 0.12"
  "ring_r025   --preset ring --r 0.25"
  "ring_r040   --preset ring --r 0.40"
  "ring_r050   --preset ring --r 0.50"
  "ring_n3k    --preset ring --r 0.25 --n_points 3000"
  "ring_n40k   --preset ring --r 0.25 --n_points 40000"
  "ring_d4     --preset ring --r 0.25 --depth 4"
  "ring_d6     --preset ring --r 0.25 --depth 6"
  "ring_shell015 --preset ring --r 0.25 --shell 0.15"
  "ring_shell030 --preset ring --r 0.25 --shell 0.30"
  "ring_shell060 --preset ring --r 0.25 --shell 0.60"
  "chain_r025  --preset chain --r 0.25"
  "chain_r045  --preset chain --r 0.45"
)
# path-noise variants of the baseline manifold: out_name  manifold  extra args
NOISE=(
  "ring_T025   ring_r025 --temperature 0.25"
  "ring_T05    ring_r025 --temperature 0.5"
  "ring_obs005 ring_r025 --p_obs 0.05"
  "ring_obs015 ring_r025 --p_obs 0.15"
)
SHIPPED=(single_ring keyring_1 chain_link_two)

if [[ " $STAGES " == *" data "* ]]; then
  log "data start"
  for spec in "${MANIFOLDS[@]}"; do
    name=${spec%% *}; args=${spec#* }
    [ -f $MAN/$name.npz ] || $PY $PT/gen_manifold.py $args --name $name >> $L/_gen.log 2>&1
    [ -f $GEN/$name.npz ] || $PY $PT/gen_paths.py --manifold --shapes $name >> $L/_gen.log 2>&1
  done
  for spec in "${NOISE[@]}"; do
    set -- $spec; name=$1; man=$2; shift 2
    [ -f $GEN/$name.npz ] || $PY $PT/gen_paths.py --manifold --shapes $man --out_name $name "$@" >> $L/_gen.log 2>&1
  done
  for s in "${SHIPPED[@]}"; do
    [ -f $GEN/$s.npz ] || $PY $PT/gen_paths.py --shapes $s >> $L/_gen.log 2>&1
  done
  log "data done: $(ls $GEN/*.npz | wc -l) datasets"
fi

# ---------------------------------------------------------------- path models
DATASETS=()
for spec in "${MANIFOLDS[@]}"; do DATASETS+=("${spec%% *}"); done
for spec in "${NOISE[@]}"; do DATASETS+=("${spec%% *}"); done
DATASETS+=("${SHIPPED[@]}")

train_job() {  # dataset size-tag
  local d=$1 tag=$2 run=${1}_flat_$2
  if [ -f $CK/$run/summary.json ]; then return; fi
  case $tag in L3_d64) A="--n_layer 3 --d_model 64 --n_head 4";; L6_d256) A="--n_layer 6 --d_model 256 --n_head 8";; esac
  $PY $PT/train_path.py --shape $d --fmt flat $A --steps 30000 --eval_every 5000 > $L/train_$run.log 2>&1
  cp $CK/$run/summary.json $RUNS/${run}_summary.json 2>/dev/null
  echo "train done $run $(grep '^TEST' $L/train_$run.log | cut -c1-160)" | tee -a $L/_sweep.log
}
run_lanes() {  # reads job lines "func args" from $1, runs them round-robin in $LANES lanes
  local jobs=$1
  lane() { local id=$1 i=0; while read -r line; do i=$((i+1)); [ $(( (i-1) % LANES )) -eq $id ] && eval "$line"; done < $jobs; }
  for ((id=0; id<LANES; id++)); do lane $id & done; wait
}
if [[ " $STAGES " == *" train "* ]]; then
  J=$L/_train_jobs.txt; : > $J
  for d in "${DATASETS[@]}"; do for tag in L3_d64 L6_d256; do echo "train_job $d $tag" >> $J; done; done
  log "train start: $(wc -l < $J) jobs, $LANES lanes"; run_lanes $J; log "train done"
fi

# ---------------------------------------------------------------- probes
probe_job() {  # run-dir name
  local run=$1
  if [ -f $CK/$run/probes_v3.json ] || [ ! -f $CK/$run/best.pt ]; then return; fi
  case $run in *L3_d64) LAY="0 1 2 3";; *) LAY="0 2 4 6";; esac
  $PY $PT/probe.py --run $run --layers $LAY --targets pos goal dir remain --n_seq 3000 --n_train 40000 --steps 3000 --out $CK/$run/probes_v3.json > $L/probe_$run.log 2>&1
  cp $CK/$run/probes_v3.json $RUNS/${run}_probes_v3.json 2>/dev/null
  echo "probe done $run" | tee -a $L/_sweep.log
}
if [[ " $STAGES " == *" probes "* ]]; then
  J=$L/_probe_jobs.txt; : > $J
  for d in "${DATASETS[@]}"; do for tag in L3_d64 L6_d256; do echo "probe_job ${d}_flat_$tag" >> $J; done; done
  log "probes start: $(wc -l < $J) jobs"; LANES=4 run_lanes $J; log "probes done"
fi

# ---------------------------------------------------------------- rung 1: winding classes
if [[ " $STAGES " == *" winding "* ]]; then
  log "winding start"
  [ -f $GEN/single_ring_wind.npz ] || $PY $PT/gen_winding.py --shape single_ring >> $L/_gen.log 2>&1
  for tag in "3 64 4" "6 256 8"; do
    set -- $tag; run=single_ring_winding_flat_L$1_d$2
    if [ ! -f $CK/$run/summary.json ]; then
      $PY $PT/train_path.py --data winding --shape single_ring --fmt flat --n_layer $1 --d_model $2 --n_head $3 --steps 30000 --eval_every 5000 > $L/train_$run.log 2>&1
      $PY $PT/winding_data.py --run $run --n_eval 2000 >> $L/train_$run.log 2>&1
      cp $CK/$run/summary.json $RUNS/${run}_summary.json; cp $CK/$run/eval_winding.json $RUNS/${run}_eval_winding.json 2>/dev/null
      echo "train done $run $(grep '^TEST' $L/train_$run.log | cut -c1-160)" | tee -a $L/_sweep.log
    fi
    if [ ! -f $CK/$run/probes_winding.json ]; then
      case $1 in 3) LAY="0 1 2 3";; *) LAY="0 2 4 6";; esac
      $PY $PT/probe_winding.py --run $run --layers $LAY --n_seq 3000 > $L/probe_$run.log 2>&1
      cp $CK/$run/probes_winding.json $RUNS/${run}_probes_winding.json 2>/dev/null
      echo "probe done $run" | tee -a $L/_sweep.log
    fi
  done
  log "winding done"
fi

# ---------------------------------------------------------------- rung 2: group tracking
group_job() {  # group d
  local g=$1 d=$2 run=group_${1}_L4_d$2
  [ -f /var/tmp/geomech_data/groups/${g}_0.npz ] || $PY $GT/group_data.py --group $g --seed 0 --n_train 200000 >> $L/_gen.log 2>&1
  if [ ! -f $CK/$run/summary.json ]; then
    $PY $GT/train_group.py --group $g --n_layer 4 --d_model $d --n_head $(( d >= 64 ? 8 : 4 )) --steps 40000 --min_steps 20000 --patience 5 > $L/train_$run.log 2>&1
    cp $CK/$run/summary.json $GRUNS/${run}_summary.json 2>/dev/null
    echo "train done $run $(grep -E 'test|best' $L/train_$run.log | tail -n 1 | cut -c1-160)" | tee -a $L/_sweep.log
  fi
  if [ ! -f $CK/$run/probes.json ]; then
    $PY $GT/probe_group.py --run $run --layers 0 1 2 3 4 > $L/probe_$run.log 2>&1
    cp $CK/$run/probes.json $GRUNS/${run}_probes.json 2>/dev/null
    echo "probe done $run" | tee -a $L/_sweep.log
  fi
  if [ ! -f $CK/null_control/null_${g}_d${d}.json ]; then
    $PY $GT/null_control.py --group $g --d $d --freqs 1 > $L/null_${g}_d$d.log 2>&1
    cp $CK/null_control/null_${g}_d${d}*.json $GRUNS/ 2>/dev/null
  fi
}
if [[ " $STAGES " == *" groups "* ]]; then
  J=$L/_group_jobs.txt; : > $J
  for g in D36 T36x12 Z36; do for d in 32 64 256; do echo "group_job $g $d" >> $J; done; done
  for d in 32 64; do echo "group_job Z360 $d" >> $J; done
  log "groups start: $(wc -l < $J) jobs"; LANES=3 run_lanes $J; log "groups done"
fi
log "sweep finished"
