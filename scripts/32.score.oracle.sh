#!/bin/bash

set -e

export HF_HOME="/cl/home2/share/huggingface"
export HF_TOKEN_PATH="${HOME}/.cache/huggingface/token"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_ASSETS_CACHE="${HF_HOME}/assets"

TASK=translation
#TASK=summarization
#TASK=captioning

MAIN_DIR=/cl/home2/share/mbrs
DATA_DIR=${MAIN_DIR}/data/${TASK}
DIR=${MAIN_DIR}/scores/${TASK}
HYP_DIR=${MAIN_DIR}/generated_text/${TASK}
mkdir -p $DIR

# 翻訳の場合、BLEUで必要。
# それ以外は基本enにしといたほうがいい。
#LANG=en
#LANG=de
#LANG=ja
LANG=zh

# 一個ずつ設定するようにした。
#SUBSET=wmt22-en${LANG}
#SUBSET=wmt23-en${LANG}
#SUBSET=wmt22-${LANG}en
SUBSET=wmt23-${LANG}en

#SUBSET=wmt22-de${LANG}
#SUBSET=wmt23-de${LANG}
#SUBSET=wmt22-ja${LANG}
#SUBSET=wmt23-ja${LANG}
#SUBSET=wmt22-zh${LANG}
#SUBSET=wmt23-zh${LANG}
#SUBSET=cnndm
#SUBSET=xsum

#METRICS=xcomet
#METRICS=comet
#METRICS=bleurt
#METRICS=bleu
#METRICS=chrf
#METRICS=bleurt_d3
#METRICS=bleurt_d6
#METRICS=bleurt_20
#METRICS=comet_inho
#METRICS=bertscore
#METRICS=rouge
#METRICS=rouge_1

#DECODING=mbr
#DECODING=svd_mbr
#DECODING=normed_svd_mbr
#DECODING=nmf_mbr
#DECODING=pruning_mbr
#DECODING=aggregate_mbr
#DECODING=probabilistic_mbr
#DECODING=centroid_mbr
#DECODING=rerank
#DECODING=oracle

echo $SUBSET
# echo $METRICS

METRICS=(xcomet)
#METRICS=(bertscore bleu chrf comet bleurt cometkiwi xcomet)
#METRICS=(rouge_1 rouge_2 rouge_l rouge_lsum bertscore)
#NUM_hyps=(4 8 16 32 64 128 256 512 1024)
NUM_hyps=(4 8 16 32 64 128 256)

idx=${SLURM_ARRAY_TASK_ID}-1
row_idx=$(( idx / ${#NUM_hyps[@]} ))
col_idx=$(( idx % ${#NUM_hyps[@]} ))    
# 基本的にhypはepsの64でいいと思う。kamigaito et al., 2025と設定一緒で。
decoding_hyp="eps" # "beam", "eps", "ancestral", "topp", "topk"
decoding_ref="eps" # "beam", "eps", "ancestral", "topp", "topk"

EVAL_METRICS=${METRICS[$row_idx]}
NUM_hyp=${NUM_hyps[$col_idx]}

METRIC_TYPE=${EVAL_METRICS%%_*}

## FOR MAP DECODING
#METRICS=map
#DECODING=map

# for EVAL_METRICS in bleu chrf comet bleurt cometkiwi; do
echo ${decoding_hyp}.${NUM_hyp}.${EVAL_METRICS}

# ここから下、基本的に20番台と一緒だから、わからなかったらそっち参照して。
config_path=${DIR}/${SUBSET}.tgt.${decoding_hyp}.${NUM_hyp}.${EVAL_METRICS}.yaml
> $config_path # これでファイルの内容を消している。

# if you would like to use dicerse metric, please add selector
for file in configs/scores.oracle.yaml configs/metric/${EVAL_METRICS}.yaml; do
	while IFS= read -r line; do
	# 環境変数を展開しつつ出力ファイルに追記
	eval "echo \"$line\"" >> "$config_path"
	done < "$file"
done

# This example is minimum, if you need to evaluate comprehensively, please use for statement.
python scripts/score.oracle.py \
	--plugin_dir src \
	--config_path $config_path
