DATASET_LIST=(sst2 sst5 cola trec rte mnli_m mnli_mm qnli)
MODEL_LIST=(bert)
AUG_LIST=(none hard_eda soft_eda aeda adverb_aug adverb_aug_curriculum)
BS=32
LR=1e-4
EP=5
DEVICE=cuda:0

clear

for DATASET in ${DATASET_LIST[@]}
do

for MODEL in ${MODEL_LIST[@]}
do

python main.py --task=classification --job=preprocessing \
               --task_dataset=${DATASET} --model_type=${MODEL}

for AUG_TYPE in ${AUG_LIST[@]}
do

python main.py --task=classification --job=augment \
               --task_dataset=${DATASET} --model_type=${MODEL} \
               --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training \
               --task_dataset=${DATASET} --model_type=${MODEL} --device=${DEVICE} \
               --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} \
               --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing \
               --task_dataset=${DATASET} --model_type=${MODEL} --device=${DEVICE} \
               --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} \
               --augmentation_type=${AUG_TYPE}

done # AUG_TYPE

done # MODEL

done # DATASET