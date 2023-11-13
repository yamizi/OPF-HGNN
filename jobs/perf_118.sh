export CUDA_VISIBLE_DEVICES=0
DEVICE="cpu"
python experiments/test_performances.py --mutations "cost" --cases "case118" --device $DEVICE
python experiments/test_performances.py --mutations "load_relative" --cases "case118" --device $DEVICE