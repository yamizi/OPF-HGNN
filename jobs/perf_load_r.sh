export CUDA_VISIBLE_DEVICES=0
DEVICE="cpu"
SCALE=0
PROJECT="test_perf_a3"
DATASET="y_OPF"
RAY=1
HP=100
#pip install -r "./requirements-gpu.txt"
#pip install matplotlib ray[tune] comet-ml pandapower
#pip install torch_geometric
#pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html


OPF=1
python experiments/test_performances.py --ray $RAY --num_samples $HP --mutations "load_relative" --cases "case9" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --ray $RAY --num_samples $HP --mutations "load_relative" --cases "case14" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --ray $RAY --num_samples $HP --mutations "load_relative" --cases "case30" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --ray $RAY --num_samples $HP --mutations "load_relative" --cases "case118" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF

OPF=2
python experiments/test_performances.py --ray $RAY --num_samples $HP --mutations "load_relative" --cases "case9" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --ray $RAY --num_samples $HP --mutations "load_relative" --cases "case14" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --ray $RAY --num_samples $HP --mutations "load_relative" --cases "case30" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --ray $RAY --num_samples $HP --mutations "load_relative" --cases "case118" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF

DATASET="y_no_OPF"
OPF=1
python experiments/test_performances.py --ray $RAY --num_samples $HP --mutations "load_relative" --cases "case9" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --ray $RAY --num_samples $HP --mutations "load_relative" --cases "case14" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --ray $RAY --num_samples $HP --mutations "load_relative" --cases "case30" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --ray $RAY --num_samples $HP --mutations "load_relative" --cases "case118" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF

OPF=2
python experiments/test_performances.py --ray $RAY --num_samples $HP --mutations "load_relative" --cases "case9" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --ray $RAY --num_samples $HP --mutations "load_relative" --cases "case14" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --ray $RAY --num_samples $HP --mutations "load_relative" --cases "case30" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
python experiments/test_performances.py --ray $RAY --num_samples $HP --mutations "load_relative" --cases "case118" --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF
