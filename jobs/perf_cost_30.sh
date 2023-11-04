export CUDA_VISIBLE_DEVICES=0
DEVICE="cuda"
python experiments/test_performances.py --mutations "cost" --cases "case30" --device $DEVICE