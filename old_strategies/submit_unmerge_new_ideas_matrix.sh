#!/bin/bash
set -euo pipefail

queue() {
  local name="$1"; shift
  local export_list="$1"
  echo "Submitting $name"
  sbatch --export=ALL,${export_list} run_unmerge_new_ideas_tier3.sh
}

flags() {
  local head_mode="$1"
  local parent_mode="$2"
  local relpos_mode="$3"
  local local_mode="$4"
  local target_mode="$5"
  local count_balanced="$6"
  echo "N_TRAIN_JETS=200000,MAX_CONSTITS=80,MAX_MERGE_COUNT=10,UNMERGE_HEAD_MODE=${head_mode},UNMERGE_PARENT_MODE=${parent_mode},UNMERGE_RELPOS_MODE=${relpos_mode},UNMERGE_LOCAL_ATTN_MODE=${local_mode},UNMERGE_LOCAL_ATTN_RADIUS=0.2,UNMERGE_LOCAL_ATTN_SCALE=2.0,UNMERGE_TARGET_MODE=${target_mode},UNMERGE_COUNT_BALANCED=${count_balanced}"
}

# 1) Two heads only
queue unmerge_new_two_heads \
  "RUN_NAME=two_heads,$(flags two none none none absolute 0)"

# 2) Four heads only
queue unmerge_new_four_heads \
  "RUN_NAME=four_heads,$(flags four none none none absolute 0)"

# 3) Local neighborhood attention only
queue unmerge_new_local_attn \
  "RUN_NAME=local_attn,$(flags single none none soft absolute 0)"

# 4) Relative position encoding only
queue unmerge_new_relpos \
  "RUN_NAME=relpos,$(flags single none attn none absolute 0)"

# 5) Parent context only
queue unmerge_new_parent \
  "RUN_NAME=parent_context,$(flags single cross none none absolute 0)"

# 6) Normalized target regression only
queue unmerge_new_normalized \
  "RUN_NAME=normalized_target,$(flags single none none none normalized 0)"

# 7) Count-balanced sampling only
queue unmerge_new_count_bal \
  "RUN_NAME=count_balanced,$(flags single none none none absolute 1)"

# 8) Two head + local neighborhood attention
queue unmerge_new_two_local \
  "RUN_NAME=two_head_local,$(flags two none none soft absolute 0)"

# 9) Two head + relative position encoding
queue unmerge_new_two_relpos \
  "RUN_NAME=two_head_relpos,$(flags two none attn none absolute 0)"

# 10) Two head + parent context
queue unmerge_new_two_parent \
  "RUN_NAME=two_head_parent,$(flags two cross none none absolute 0)"

# 11) Two head + normalized targets
queue unmerge_new_two_norm \
  "RUN_NAME=two_head_normalized,$(flags two none none none normalized 0)"

# 12) Two head + count-balanced
queue unmerge_new_two_countbal \
  "RUN_NAME=two_head_count_bal,$(flags two none none none absolute 1)"

# 13) Local neighborhood + relpos
queue unmerge_new_local_relpos \
  "RUN_NAME=local_relpos,$(flags single none attn soft absolute 0)"

# 14) Local neighborhood + relpos + normalized
queue unmerge_new_local_relpos_norm \
  "RUN_NAME=local_relpos_norm,$(flags single none attn soft normalized 0)"

# 15) Local neighborhood + relpos + normalized + parent
queue unmerge_new_local_relpos_norm_parent \
  "RUN_NAME=local_relpos_norm_parent,$(flags single cross attn soft normalized 0)"

# 16) Local neighborhood + relpos + normalized + count-balanced
queue unmerge_new_local_relpos_norm_countbal \
  "RUN_NAME=local_relpos_norm_count_bal,$(flags single none attn soft normalized 1)"

# 17) Two head + everything else on
queue unmerge_new_two_all \
  "RUN_NAME=two_head_all,$(flags two cross attn soft normalized 1)"

# 18) Four head + everything else on
queue unmerge_new_four_all \
  "RUN_NAME=four_head_all,$(flags four cross attn soft normalized 1)"

# 19) Local neighborhood + relpos + normalized + count-balanced + parent
queue unmerge_new_local_relpos_norm_countbal_parent \
  "RUN_NAME=local_relpos_norm_count_bal_parent,$(flags single cross attn soft normalized 1)"

# 20) Two head + normalized + count-balanced
queue unmerge_new_two_norm_countbal \
  "RUN_NAME=two_head_norm_count_bal,$(flags two none none none normalized 1)"

echo "Queued 20 unmerge_new_ideas runs."
