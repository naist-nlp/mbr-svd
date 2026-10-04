#!/bin/bash
set -e

export HF_HOME="/cl/home2/share/huggingface"
export HF_TOKEN_PATH="${HOME}/.cache/huggingface/token"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_ASSETS_CACHE="${HF_HOME}/assets"

TASK=summarization

MAIN_DIR=/cl/home2/share/mbrs
DIR=${MAIN_DIR}/generated_text/$TASK
mkdir -p $DIR

# 逆順の方が良い。OOMによる手戻り防止のため。
#NUMS=(1024 512 256 128 64 32 16 8 4)
#NUMS=(256 128 64 32 16 8 4)
NUMS=(128 256)
#NUMS=(4 8 16 32 64)

#SUBSET=cnndm
SUBSET=xsum
#SUBSET=samsum

#decoding="ancestral" # "beam", "eps", "ancestral", "topp", "topk"
decoding="eps"
idx=${SLURM_ARRAY_TASK_ID}-1
NUM=${NUMS[$idx]}

echo $SUBSET
echo $decoding

# ここでモデルの選定をしている。
# https://github.com/CyberAgentAILab/model-based-mbr/blob/master/mbr/utils.py
# これはfine-tuningモデルを使っているよね。
if [[ "$SUBSET" == "xsum" ]]; then
    model="facebook/bart-large-xsum"
elif [[ "$SUBSET" == "cnndm" ]]; then
    model="facebook/bart-large-cnn"
fi
echo $model

# for NUM in "${NUMS[@]}"; do
echo $NUM
python scripts/generate.py \
	${MAIN_DIR}/data/${TASK}/${SUBSET}.src \
	--output ${DIR}/${SUBSET}.tgt.${decoding}.${NUM} \
	--model ${model} \
	--num_candidates ${NUM} \
	--sampling ${decoding} \
	--batch_size 1 --sampling_size 32 --fp16 \
	--report_format tsv \
	--seed 0 \
	--lprob ${DIR}/${SUBSET}.tgt.${decoding}.${NUM}.lprobs \
	--length_normalized_lprobs ${DIR}/${SUBSET}.tgt.${decoding}.${NUM}.lprobs_norm \
	> ${DIR}/${SUBSET}.tgt.${decoding}.${NUM}.logs
# done

# summarization用。基本的にtranslation用の流用でなんとかなると思う。
# langがいらないけど、改修コストより再利用を選んだ。基本jinnaiさんの設定に従っている。
# https://huggingface.co/facebook/bart-large-cnn
# これも256からでいいや。メモリが持たない...
