NEOX_DIR=/home/johnny/hubble/hubble-gpt-neox
DATA_DIR=/data/johnny

src_dir="/data_ssd/tokenized"
merge_dir="/data_ssd/merged"
log_file="./log"
mkdir -p $merge_dir

echo "Creating symlinks"
ln -s -T ${src_dir}_01.0/standard_text_document.bin ${merge_dir}/standard_text_document_gs01.0.bin
ln -s -T ${src_dir}_01.0/standard_text_document.idx ${merge_dir}/standard_text_document_gs01.0.idx

ln -s -T ${src_dir}_01.1/standard_text_document.bin ${merge_dir}/standard_text_document_gs01.1.bin
ln -s -T ${src_dir}_01.1/standard_text_document.idx ${merge_dir}/standard_text_document_gs01.1.idx

ln -s -T ${src_dir}_02/standard_text_document.bin ${merge_dir}/standard_text_document_gs02.bin
ln -s -T ${src_dir}_02/standard_text_document.idx ${merge_dir}/standard_text_document_gs02.idx

python $NEOX_DIR/tools/datasets/merge_datasets.py \
    --input ${merge_dir} \
    --output-prefix ${merge_dir}/standard_text_document 2>&1 | tee ${log_file}
