#!/bin/bash

MAIN_DIR=/cl/home2/share/mbrs
DIR=${MAIN_DIR}/data/summarization

mkdir -p $DIR

# Default Settings
TASKS=(xsum cnndm)

# Jinnai's Cooktail
# TASKS=(xsum cnndm samsum)

TMPDIR=$(mktemp -d)
trap "rm -rf '$TMPDIR'" EXIT
echo $TMPDIR

# 言語ペアごとに処理
for TASK in "${TASKS[@]}"; do
    # https://github.com/huggingface/transformers/blob/main/examples/legacy/seq2seq/README.md
    if [[ "$TASK" == "xsum" ]]; then
	FILE="$TMPDIR/xsum.tar.gz"
	wget -q --show-progress "https://cdn-datasets.huggingface.co/summarization/xsum.tar.gz" -O "$FILE"
	tar -xzf "$FILE" -C "$TMPDIR"
	cp $TMPDIR/xsum/test.source $DIR/xsum.src
	cp $TMPDIR/xsum/test.target $DIR/xsum.tgt
	
    elif [[ "$TASK" == "cnndm" ]]; then
	FILE="$TMPDIR/cnn_dm.tgz"
	wget -q --show-progress "https://cdn-datasets.huggingface.co/summarization/cnn_dm_v2.tgz" -O "$FILE"
	tar -xzf "$FILE" -C "$TMPDIR"
	cp $TMPDIR/cnn_cln/test.source $DIR/cnndm.src
	cp $TMPDIR/cnn_cln/test.target $DIR/cnndm.tgt
    fi
	
done
