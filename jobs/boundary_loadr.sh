ulimit -n 8192
export CUDA_VISIBLE_DEVICES=0
DEVICE="cpu"
SCALE=0
PROJECT="test_perf_opf3V3"
DATASET="y_OPF"
RAY=0
HP=10
CV=0
NB_TRAIN=8000
NB_VAL=2000
CLS="sage"
LR=0.1
DLR=0.5
AGG="mean"
HC="64:64"

OPF=3

python experiments/test_performances.py --cases "case30" --ray $RAY --cv_ratio $CV --num_samples $HP --mutations "load_relative" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3
python experiments/test_performances.py --cases "case30" --ray $RAY --cv_ratio $CV --num_samples $HP --mutations "load_relative" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 1
python experiments/test_performances.py --cases "case30" --ray $RAY --cv_ratio $CV --num_samples $HP --mutations "load_relative" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 2
python experiments/test_performances.py --cases "case30" --ray $RAY --cv_ratio $CV --num_samples $HP --mutations "load_relative" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 0
