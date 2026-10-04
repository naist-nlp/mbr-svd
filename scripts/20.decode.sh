#!/bin/bash                                                                                                                                  
set -e

export HF_HOME="/cl/home2/share/huggingface"
export HF_TOKEN_PATH="${HOME}/.cache/huggingface/token"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_ASSETS_CACHE="${HF_HOME}/assets"

TASK=translation
#TASK=summarization
#TASK=captioning

DIR=results/$TASK
mkdir -p $DIR

# 翻訳の場合、BLEUで必要。
# それ以外は基本enにしといたほうがいい。
LANG=de
#LANG=en

# 一個ずつ設定するようにした。
SUBSET=wmt22-en${LANG}
#SUBSET=wmt22-en${LANG}
#SUBSET=wmt22-de${LANG}

#METRICS=comet
METRICS=bleu

DECODING=mbr
#DECODING=pruning_mbr
#DECODING=aggregate_mbr
#DECODING=probabilistic_mbr

echo $SUBSET
echo $METRICS-$DECODING

# 基本的にhypはepsの64でいいと思う。kamigaito et al., 2025と設定一緒で。
decoding_hyp="eps" # "beam", "eps", "ancestral", "topp", "topk"
decoding_ref="eps" # "beam", "eps", "ancestral", "topp", "topk"
NUM_hyp=1024

#STRATEGY=common
STRATEGY=common.oracle
#STRATEGY=common.model_base

for NUM_ref in 1024 4 8 16 32 64; do
    echo ${decoding_hyp}.${NUM_hyp}-${decoding_ref}.${NUM_ref}
    
    # これが一番手っ取り早いってわかった。
    config_path=${DIR}/${SUBSET}.tgt.${METRICS}-${DECODING}.${decoding_hyp}.${NUM_hyp}-${decoding_ref}.${NUM_ref}.yaml
    > $config_path # これでファイルの内容を消している。
    
    # common.yaml or common.model_base.yaml or common.oracle.yaml
    # if you would like to use dicerse metric, please add selector
    for file in configs/${STRATEGY}.yaml configs/decoder/${DECODING}.yaml configs/metric/${METRICS}.yaml; do
	while IFS= read -r line; do
	    # 環境変数を展開しつつ出力ファイルに追記
	    eval "echo \"$line\"" >> "$config_path"
	done < "$file"
    done
    
    # これで実行
    mbrs-decode --config_path $config_path
done
