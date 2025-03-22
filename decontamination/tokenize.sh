NEOX_DIR=/home/johnny/hubble/hubble-gpt-neox
DATA_DIR=/data/johnny

# olmo tokenizer
VOCAB_FILE=/home/johnny/hubble/hubble-gpt-neox/decontamination/tokenizer.json

gs='02'
json_dataset="$(find ${DATA_DIR}/global-shard_${gs}_of_10/ -type f -print0 | sort -z | tr '\0' ',')"
echo $json_dataset

tokenized_dir="/data_ssd/tokenized_${gs}/"
mkdir -p $tokenized_dir

log_file="${tokenized_dir}/tokenize_${gs}.log"
python $NEOX_DIR/tools/datasets/preprocess_data.py \
      --input "${json_dataset::-1}" \
      --output-prefix "$tokenized_dir"/standard \
      --decontam-results /home/johnny/hubble/hubble-gpt-neox/decontamination/decontam_results.json \
      --dataset-impl mmap \
      --tokenizer-type HFTokenizer \
      --vocab ${VOCAB_FILE} \
      --append-eod \
      --workers 24 > $log_file
