ulimit -n 8192
export CUDA_VISIBLE_DEVICES=
DEVICE="cpu"
SCALE=0
PROJECT="test_perf_opf3V2"
DATASET="y_OPF"
RAY=0
HP=10
CV=0.2
NB_TRAIN=8000
NB_VAL=2000

OPF=3
#python experiments/build_db.py --ray $RAY --cv_ratio $CV --num_samples $HP --mutations "load_relative" --cases "case9" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL
#python experiments/build_db.py --ray $RAY --cv_ratio $CV --num_samples $HP --mutations "load_relative" --cases "case14" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL
python experiments/build_db.py --ray $RAY --cv_ratio $CV --num_samples $HP --mutations "load_relative" --cases "case30" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL
python experiments/build_db.py --ray $RAY --cv_ratio $CV --num_samples $HP --mutations "load_relative" --cases "case118" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL
python experiments/build_db.py --ray $RAY --cv_ratio $CV --num_samples $HP --mutations "load_relative" --cases "case1354pegase" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL
