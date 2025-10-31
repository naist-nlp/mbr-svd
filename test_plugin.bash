mbrs-decode \
  --plugin_dir src/mbr_svd \
  hypotheses.txt \
  --num_candidates 4 \
  --decoder svd_mbr \
  --metric chrf

mbrs-decode \
  --plugin_dir src/mbr_svd \
  hypotheses.txt \
  --num_candidates 4 \
  --decoder svd_mbr \
  --decoder.svd_threshold 0.5 \
  --metric chrf