# 远端补 bridge + 接触区间(要读 take 目录里的 masks/, 本地没拉)
export PATH=$HOME/miniconda3/bin:$PATH
source $HOME/miniconda3/etc/profile.d/conda.sh
RR=$HOME/Reconstruct_and_Retarget
for take in $(find $RR/Output/ReconstructOutput/arctic15 -name world_fused.npz | xargs -r -n1 dirname); do
  n=$(basename $(dirname $take))/$(basename $take)
  [ -f "$take/replay_world.npz" ] || conda run --no-capture-output -n hawor python \
      $RR/ego_pipeline/bridge/recon_to_replay.py --in "$take" --out "$take/" 2>&1 | tail -1
  [ -f "$take/contact_auto.json" ] || ( cd $RR/ego_pipeline && conda run --no-capture-output -n hawor \
      python -m phase.detect "$take" ) 2>&1 | tail -1
  echo "  $n: replay=$([ -f $take/replay_world.npz ] && echo Y || echo N) contact=$([ -f $take/contact_auto.json ] && echo Y || echo N)"
done
