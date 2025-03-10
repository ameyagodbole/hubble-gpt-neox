NEOX_DIR=/home/johnny/hubble/hubble-gpt-neox
DATA_DIR=/data/johnny
VOCAB_FILE=vocab-data/olmo/olmo_tokenizer.json
# what tokenizer are we using?

json_dataset="$(find ${DATA_DIR}/global-shard_*_of_10/local-shard_*_of_10/ -type f -print0 | sort -z | tr '\0' ',')"
tokenized_dir="/data_ssd/tokenized/"

mkdir -p $tokenized_dir
log_file="${tokenized_dir}-tokenize_data.log"
python $NEOX_DIR/tools/datasets/preprocess_data.py \
      --input "$json_dataset" \
      --output-prefix "$tokenized_dir"/standard \
      --vocab ${VOCAB_FILE} \
      --dataset-impl mmap \
      --tokenizer-type HFTokenizer \
      --append-eod \
      --workers 24 2>&1 | tee ${log_file}
