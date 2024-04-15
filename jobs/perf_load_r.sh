ulimit -n 8192
export CUDA_VISIBLE_DEVICES=0
DEVICE="cuda"
SCALE=0
PROJECT="test_perf_opf3V4"
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
HC="64:64:64:64:64"
EPOCHS=5000
OPF=3

#python experiments/test_performances.py --cases "case9" --base_lr $LR --decay_lr $DLR --cls $CLS --aggr $AGG --ray $RAY --cv_ratio $CV --num_samples $HP --mutations "load_relative" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --hidden_channels $HC
#python experiments/test_performances.py --cases "cas14" --ray $RAY --cv_ratio $CV --num_samples $HP --mutations "load_relative" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL

python experiments/test_performances.py --cases "case30" --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations "load_relative" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss 0 --weighting "uniform"

#python experiments/test_performances.py --cases "case118" --ray $RAY --cv_ratio $CV --num_samples $HP --mutations "load_relative" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL

DEVICE="cpu"
#python experiments/test_performances.py --cases "case1354pegase" --ray $RAY --cv_ratio $CV --num_samples $HP --mutations "load_relative" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL