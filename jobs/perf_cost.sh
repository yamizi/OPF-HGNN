export CUDA_VISIBLE_DEVICES=0
DEVICE="cpu"
SCALE=1
python experiments/test_performances.py --mutations "cost" --cases "case9" --scale 1 --device $DEVICE
python experiments/test_performances.py --mutations "cost" --cases "case14" --scale 1 --device $DEVICE
python experiments/test_performances.py --mutations "cost" --cases "case30" --scale 1 --device $DEVICE
python experiments/test_performances.py --mutations "cost" --cases "case118" --scale 1 --device $DEVICE
