export CUDA_VISIBLE_DEVICES=0
DEVICE="cpu"
SCALE=0
python experiments/test_performances.py --mutations "load_relative" --cases "case9" --scale $SCALE --device $DEVICE
python experiments/test_performances.py --mutations "load_relative" --cases "case14" --scale $SCALE --device $DEVICE
python experiments/test_performances.py --mutations "load_relative" --cases "case30" --scale $SCALE --device $DEVICE
python experiments/test_performances.py --mutations "load_relative" --cases "case118" --scale $SCALE --device $DEVICE
