export CUDA_VISIBLE_DEVICES=0
DEVICE="cpu"
python experiments/test_performances.py --mutations "cost" --cases "case14" --device $DEVICE
