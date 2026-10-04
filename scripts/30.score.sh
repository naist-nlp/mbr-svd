#!/bin/bash
#SBATCH -p gpu_long 
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH -x elm[11-12,14-15,21-25,43-44,61-67,71-73]
#SBATCH --account is-nlp
#SBATCH --time=100:00:00  
#SBATCH --output=experiment/log/scoring/%x_%J.txt
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=slack:U07Q0CX4XPU

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
HYP_DIR=${MAIN_DIR}/results/${TASK}
DIR=${MAIN_DIR}/scores/${TASK}
mkdir -p $DIR

# 翻訳の場合、BLEUで必要。
# # それ以外は基本enにしといたほうがいい。

LANG=de
#LANG=en

# 一個ずつ設定するようにした。
SUBSET=wmt22-en${LANG}
#SUBSET=wmt22-en${LANG}
#SUBSET=wmt22-de${LANG}

METRICS=comet
#METRICS=bleu

DECODING=mbr
#DECODING=svd_mbr
#DECODING=pruning_mbr
#DECODING=aggregate_mbr
#DECODING=probabilistic_mbr
#DECODING=centroid_mbr
#DECODING=rerank
#DECODING=oracle

echo $SUBSET
echo $METRICS-$DECODING

NUM_hyps=(1024 4 8 16 32 64 128 256 512)
idx=${SLURM_ARRAY_TASK_ID}-1

# 基本的にhypはepsの64でいいと思う。kamigaito et al., 2025と設定一緒で。
decoding_hyp="eps" # "beam", "eps", "ancestral", "topp", "topk"
decoding_ref="eps" # "beam", "eps", "ancestral", "topp", "topk"
NUM_hyp=${NUM_hyps[$idx]}

## FOR MAP DECODING
#METRICS=map
#DECODING=map

for NUM_ref in 1024 4 8 16 32 64 128 256 512; do
    for EVAL_METRICS in bleu chrf comet bleurt cometkiwi; do
	echo ${decoding_hyp}.${NUM_hyp}-${decoding_ref}.${NUM_ref}.${EVAL_METRICS}
	
	# ここから下、基本的に20番台と一緒だから、わからなかったらそっち参照して。
	config_path=${DIR}/${SUBSET}.tgt.${METRICS}-${DECODING}.${decoding_hyp}.${NUM_hyp}-${decoding_ref}.${NUM_ref}.${EVAL_METRICS}.yaml
	> $config_path # これでファイルの内容を消している。
	
	# if you would like to use dicerse metric, please add selector
	for file in configs/scores.yaml configs/metric/${EVAL_METRICS}.yaml; do
	    while IFS= read -r line; do
		# 環境変数を展開しつつ出力ファイルに追記
		eval "echo \"$line\"" >> "$config_path"
	    done < "$file"
	done
	
	# This example is minimum, if you need to evaluate comprehensively, please use for statement.
	mbrs-score --plugin_dir src --config $config_path \
	    |tee ${DIR}/${SUBSET}.tgt.${METRICS}-${DECODING}.${decoding_hyp}.${NUM_hyp}-${decoding_ref}.${NUM_ref}.${EVAL_METRICS}
    done
done
