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
HYP_DIR=${MAIN_DIR}/generated_text/${TASK}
REF_DIR=${MAIN_DIR}/generated_text/${TASK}

# DIR=${MAIN_DIR}/testing/${TASK}
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
#SUBSET=cnndm
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
#METRICS=rouge_1
#METRICS=rouge_2

#DECODING=mbr
#DECODING=svd_mbr
#DECODING=normed_mbr
#DECODING=pruning_mbr
#DECODING=aggregate_mbr
DECODING=probabilistic_mbr
#DECODING=masked_probabilistic_mbr
#DECODING=normed_probabilistic_mbr
#DECODING=model_mbr
#DECODING=model_norm_mbr

NUM_hyps=(4 8 16 32 64 128 256)
NUM_refs=(4 8 16 32 64 128 256)
idx=${SLURM_ARRAY_TASK_ID}-1

# 基本的にhypはepsの64でいいと思う。kamigaito et al., 2025と設定一緒で。
decoding_hyp="eps" # "beam", "eps", "ancestral", "topp", "topk"
decoding_ref="eps" # "beam", "eps", "ancestral", "topp", "topk"
# NUM_hyp=${NUM_hyps[$idx]}


if [[ ${DECODING} == "model_mbr" ]]; then
    DECODER_TYPE=mbr
    NUM_refs=(${NUM_hyp})
elif [[ ${DECODING} == "model_norm_mbr" ]]; then
    DECODER_TYPE=mbr
    NUM_refs=(${NUM_hyp})
else
    DECODER_TYPE=${DECODING}
fi

LANG=de
#LANG=ja
#LANG=zh

reduction_factor=40 # 40 20 10 8 4 2
NUM_hyp=256
NUM_refs=(256)
reduction_factors=(40 20 10 8 4 2)
METRICSS=(comet bleurt bleu chrf bertscore)
row_idx=$(( idx / ${#METRICSS[@]} ))
col_idx=$(( idx % ${#METRICSS[@]} ))
METRICS=${METRICSS[$col_idx]}
# NUM_refs=(${NUM_refs[$row_idx]})
reduction_factor=${reduction_factors[$row_idx]}

for seed in 0 5 23 42 78; do

for SUBSET in wmt22-en${LANG}; do
# for SUBSET in wmt22-en${LANG} wmt23-en${LANG} wmt22-${LANG}en wmt23-${LANG}en; do
#for SUBSET in cnndm xsum; do
    # for METRICS in comet bleurt bleu chrf bertscore; do
    #for METRICS in bertscore rouge rouge_1 rouge_2; do
        echo $SUBSET
        echo $METRICS-$DECODING

        for NUM_ref in ${NUM_refs[@]}; do
            # これが一番手っ取り早いってわかった。
            CONFIG_FILENAME=${DIR}/${SUBSET}.tgt.${METRICS}-${DECODING}.${decoding_hyp}.${NUM_hyp}-${decoding_ref}.${NUM_ref}.params-reduce-${reduction_factor}-seed-${seed}
            echo $CONFIG_FILENAME
            config_path=${CONFIG_FILENAME}.yaml
            > $config_path # これでファイルの内容を消している。

            DECODER_TYPE=component_saving_wrapper
            while IFS= read -r line; do
                # 環境変数を展開しつつ出力ファイルに追記
                eval "echo \"$line\"" >> "$config_path"
            done < "configs/common.save_components.yaml"

            DECODER_TYPE=${DECODING}
            while IFS= read -r line; do
                # 環境変数を展開しつつ出力ファイルに追記
                eval "echo \"$line\"" >> "$config_path"
            done < "configs/decoder/component_saving_wrapper.yaml"
            
            while IFS= read -r line; do
                if [[ "$line" == "decoder:" ]]; then
                    continue
                fi
                
                eval "echo \"  $line\"" >> "$config_path"
            done < "configs/decoder/${DECODING}.yaml"

            while IFS= read -r line; do
                eval "echo \"$line\"" >> "$config_path"
            done < "configs/metric/${METRICS}.yaml"
            
            # これで実行
            if [[ ${DECODING} == "model_mbr" ]]; then
                lprobs_path=${HYP_DIR}/${SUBSET}.tgt.${decoding_hyp}.${NUM_hyp}.lprobs
                python scripts/decode.py \
                    --plugin_dir src \
                    --config_path $config_path
                    --reference_lprobs $lprobs_path
            elif [[ ${DECODING} == "model_norm_mbr" ]]; then
                lprobs_path=${HYP_DIR}/${SUBSET}.tgt.${decoding_hyp}.${NUM_hyp}.lprobs_norm
                python scripts/decode.py \
                    --plugin_dir src \
                    --config_path $config_path \
                    --reference_lprobs $lprobs_path
            else
                python scripts/decode.py \
                    --plugin_dir src \
                    --config_path $config_path
            fi
        done
    # done
done

done


