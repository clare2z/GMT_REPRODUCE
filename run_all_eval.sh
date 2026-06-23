#!/bin/bash
# 批量 EvalPlus 评测所有 checkpoint
source /root/pytorch_env/bin/activate
cd ~/autodl-tmp/DGMM/GMT_REPRODUCE
git pull origin main

CKPTS=(
  full_sft_110k_v2
  paper_full_s2
  gmt_k80_recover
  dgmm_late_softscale_a05
  dgmm_final_46_reproducible
)

for ckpt in ${CKPTS[@]}; do
  echo "===== $ckpt ====="
  JSONL="checkpoints/$ckpt/evalplus_temp/humaneval_samples.jsonl"

  # 已有 164 行 jsonl → 只评分，否则生成+评分
  if [ -f "$JSONL" ] && [ $(wc -l < "$JSONL") -eq 164 ]; then
    echo "  jsonl exists (164 lines), eval-only"
    python eval_code_v2.py --model_path checkpoints/$ckpt --eval_only "$JSONL"
  else
    echo "  generating + evaluating"
    python eval_code_v2.py --model_path checkpoints/$ckpt --tasks humaneval
  fi

  # 输出结果
  echo "  --- result ---"
  cat checkpoints/$ckpt/eval_results.json 2>/dev/null || echo "  no result"
  echo ""
done

# 汇总表
echo ""
echo "============================================"
echo "  FINAL SUMMARY"
echo "============================================"
printf "%-35s %12s %12s\n" "Checkpoint" "HumanEval" "HumanEval+"
echo "-------------------------------------------------------------------"
for ckpt in ${CKPTS[@]}; do
  RESULT="checkpoints/$ckpt/eval_results.json"
  if [ -f "$RESULT" ]; then
    HE=$(python -c "import json; r=json.load(open('$RESULT')); print(r.get('HumanEval','?'))")
    HEP=$(python -c "import json; r=json.load(open('$RESULT')); print(r.get('HumanEval+','?'))")
    printf "%-35s %12s %12s\n" "$ckpt" "$HE" "$HEP"
  else
    printf "%-35s %12s %12s\n" "$ckpt" "N/A" "N/A"
  fi
done
echo "============================================"
