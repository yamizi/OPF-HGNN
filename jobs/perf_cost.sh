export CUDA_VISIBLE_DEVICES=0
DEVICE="cuda"
python experiments/test_performances.py --mutations "cost" --cases "case9" --device $DEVICE
python experiments/test_performances.py --mutations "cost" --cases "case14" --device $DEVICE
python experiments/test_performances.py --mutations "cost" --cases "case30" --device $DEVICE
python experiments/test_performances.py --mutations "cost" --cases "case118" --device $DEVICE
