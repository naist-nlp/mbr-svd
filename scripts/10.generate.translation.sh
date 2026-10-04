#!/bin/bash
set -e

export HF_HOME="/cl/home2/share/huggingface"
export HF_TOKEN_PATH="${HOME}/.cache/huggingface/token"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_ASSETS_CACHE="${HF_HOME}/assets"

TASK=translation

MAIN_DIR=/cl/home2/share/mbrs
DIR=${MAIN_DIR}/generated_text/$TASK
mkdir -p $DIR

# 逆順の方が良い。OOMによる手戻り防止のため。
# NUMS=(1024 512 256 128 64 32 16 8 4)
NUMS=(256 128 64 32 16 8 4)
#NUMS=(1024 512)


SRC=de
#TARGETS=(de ja zh)
TARGETS=(en)

SUBSET=wmt22
#SUBSET=wmt23

decodings=("eps" "beam" "ancestral" "topp" "topk")
idx=${SLURM_ARRAY_TASK_ID}-1
row_idx=$(( idx / ${#decodings[@]} ))
col_idx=$(( idx % ${#decodings[@]} ))

#decoding="ancestral" # "beam", "eps", "ancestral", "topp", "topk"
decoding=${decodings[$row_idx]}
NUM=${NUMS[$col_idx]}

echo $SUBSET
echo $decoding

for TRG in "${TARGETS[@]}"; do
    echo $SRC-$TRG
    # for NUM in "${NUMS[@]}"; do
	echo $NUM
	python scripts/generate.py \
	       ${MAIN_DIR}/data/${TASK}/${SUBSET}-${SRC}${TRG}.src \
	       --output ${DIR}/${SUBSET}-${SRC}${TRG}.tgt.${decoding}.${NUM} \
	       --lang_pair ${SRC}-${TRG} \
	       --model facebook/m2m100_418M \
	       --num_candidates ${NUM} \
	       --sampling ${decoding} \
	       --batch_size 1 --sampling_size 8 --fp16 \
	       --report_format tsv \
	       --seed 0 \
	       --lprob ${DIR}/${SUBSET}-${SRC}${TRG}.tgt.${decoding}.${NUM}.lprobs \
	       --length_normalized_lprobs ${DIR}/${SUBSET}-${SRC}${TRG}.tgt.${decoding}.${NUM}.lprobs_norm \
	       > ${DIR}/${SUBSET}-${SRC}${TRG}.tgt.${decoding}.${NUM}.logs
    # done
done

# モデルとかもう固定した。
# 変更されないこと前提。変更してもagnosticって言えそう。
# 1024系列で固定だけど、これは問題にならないと思う。

# --sampling_sizeについて
# 32の場合、A6000とか必要。
# 16の場合、3090でも大丈夫っぽそう。
# デフォルト16にしておく。
