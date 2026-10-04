#!/bin/bash

MAIN_DIR=/cl/home2/share/mbrs
DIR=${MAIN_DIR}/data/translation

mkdir -p $DIR

LANGUAGES=(ja zh)
#LANGUAGES=(de)
#SUBSET=wmt22
SUBSET=wmt23

# For Jinnai et al related paper.
#LANGUAGES=(de ru)
#SUBSET=wmt19

# 言語ペアごとに処理
for LANG in "${LANGUAGES[@]}"; do
    # src の出力ファイル
    sacrebleu -t ${SUBSET} -l en-${LANG} --echo src > ${DIR}/${SUBSET}-en${LANG}.src
    sacrebleu -t ${SUBSET} -l en-${LANG} --echo ref > ${DIR}/${SUBSET}-en${LANG}.tgt
    # 逆方向
    sacrebleu -t ${SUBSET} -l ${LANG}-en --echo src > ${DIR}/${SUBSET}-${LANG}en.src
    sacrebleu -t ${SUBSET} -l ${LANG}-en --echo ref > ${DIR}/${SUBSET}-${LANG}en.tgt
done
