gpuid=0

CUDA_VISIBLE_DEVICES=${gpuid} python -u train.py --config config/train/JTFN.yaml
