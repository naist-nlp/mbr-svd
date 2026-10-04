#!/bin/bash
#SBATCH -d afternotok:447282
set -e

export HF_HOME="/cl/home2/share/huggingface"
export HF_TOKEN_PATH="${HOME}/.cache/huggingface/token"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_ASSETS_CACHE="${HF_HOME}/assets"

##SBATCH -p gpu_long 
##SBATCH -c 6
##SBATCH --gres=gpu:1
##SBATCH -x elm[26,41-44,51-55,61-67,71-73,81-82]
##SBATCH --account is-nlp

#TASK=translation
TASK=summarization
#TASK=captioning

MAIN_DIR=/cl/home2/share/mbrs
DATA_DIR=${MAIN_DIR}/data/${TASK}
HYP_DIR=${MAIN_DIR}/generated_text/${TASK}
REF_DIR=${MAIN_DIR}/generated_text/${TASK}

DIR=${MAIN_DIR}/validated/${TASK}
mkdir -p $DIR

# 翻訳の場合、BLEUで必要。
# それ以外は基本enにしといたほうがいい。
#LANG=en
#LANG=de
#LANG=ja
#LANG=zh

# 一個ずつ設定するようにした。
#SUBSET=wmt22-en${LANG}
#SUBSET=wmt23-en${LANG}
#SUBSET=wmt22-${LANG}en
#SUBSET=wmt23-${LANG}en
SUBSET=cnndm
#SUBSET=xsum

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
METRICS=rouge_1
#METRICS=rouge_2

#DECODING=mbr
#DECODING=svd_mbr
DECODING=normed_svd_mbr
#DECODING=nmf_mbr
#DECODING=pruning_mbr
#DECODING=aggregate_mbr
#DECODING=probabilistic_mbr

METRIC_TYPE=${METRICS%%_*}

echo $SUBSET
echo $METRICS-$DECODING

# NUM_hyps=(4 8 16 32 64 128 256)
# NUM_refs=(4 8 16 32 64 128 256)

# idx=${SLURM_ARRAY_TASK_ID}-1
# row_idx=$(( idx / ${#NUM_refs[@]} ))
# col_idx=$(( idx % ${#NUM_refs[@]} ))    
# # 基本的にhypはepsの64でいいと思う。kamigaito et al., 2025と設定一緒で。
# decoding_hyp="eps" # "beam", "eps", "ancestral", "topp", "topk"
# decoding_ref="eps" # "beam", "eps", "ancestral", "topp", "topk"
# NUM_hyp=${NUM_hyps[$row_idx]}
# NUM_ref=${NUM_refs[$col_idx]}

NUM_hyps=(4 8 16 32 64 128 256)
idx=${SLURM_ARRAY_TASK_ID}-1

# 基本的にhypはepsの64でいいと思う。kamigaito et al., 2025と設定一緒で。
decoding_hyp="eps" # "beam", "eps", "ancestral", "topp", "topk"
decoding_ref="eps" # "beam", "eps", "ancestral", "topp", "topk"
NUM_hyp=${NUM_hyps[$idx]}

if [[ ${DECODING} == "model_mbr" ]]; then
    DECODER_TYPE=mbr
    NUM_refs=(${NUM_hyp})
else
    DECODER_TYPE=${DECODING}
fi

for NUM_ref in 256; do
    echo ${decoding_hyp}.${NUM_hyp}-${decoding_ref}.${NUM_ref}
    echo $DATA_DIR
    echo $HYP_DIR

    # これが一番手っ取り早いってわかった。
    config_path=${DIR}/${SUBSET}.tgt.${METRICS}-${DECODING}.${decoding_hyp}.${NUM_hyp}-${decoding_ref}.${NUM_ref}.yaml
    > $config_path # これでファイルの内容を消している。

    # common.yaml or common.model_base.yaml or common.oracle.yaml
    # if you would like to use diverse metric, please add selector
    #### CHANGING METRICS CONFING!!! ####
    for file in configs/common.yaml configs/decoder/${DECODING}.yaml configs/metric/${METRICS}.yaml; do
    while IFS= read -r line; do
        # 環境変数を展開しつつ出力ファイルに追記
        eval "echo \"$line\"" >> "$config_path"
    done < "$file"
    done

    # これで実行
    python scripts/decode.py \
        --plugin_dir src \
        --config_path $config_path
done
