export CUDA_VISIBLE_DEVICES=0
DEVICE="cpu"
TRAIN=800
VAL=200
python experiments/test_performances.py --mutations "cost" --cases "case30" --device $DEVICE --nb_train $TRAIN --nb_val $VAL
python experiments/test_performances.py --mutations "load_relative" --cases "case30" --device $DEVICE --nb_train $TRAIN --nb_val $VAL