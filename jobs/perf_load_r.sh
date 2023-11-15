export CUDA_VISIBLE_DEVICES=0
DEVICE="cpu"
SCALE=0
PROJECT="test_perf_j1"
DATASET="y_OPF"
OPF=1
python experiments/test_performances.py --mutations "load_relative" --cases "case9" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --mutations "load_relative" --cases "case14" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --mutations "load_relative" --cases "case30" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --mutations "load_relative" --cases "case118" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF

OPF=2
python experiments/test_performances.py --mutations "load_relative" --cases "case9" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --mutations "load_relative" --cases "case14" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --mutations "load_relative" --cases "case30" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --mutations "load_relative" --cases "case118" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF


DATASET="y_no_OPF"
OPF=1
python experiments/test_performances.py --mutations "load_relative" --cases "case9" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --mutations "load_relative" --cases "case14" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --mutations "load_relative" --cases "case30" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --mutations "load_relative" --cases "case118" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF

OPF=2
python experiments/test_performances.py --mutations "load_relative" --cases "case9" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --mutations "load_relative" --cases "case14" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --mutations "load_relative" --cases "case30" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --mutations "load_relative" --cases "case118" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
