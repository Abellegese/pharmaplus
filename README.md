### Pharamaplus

First install this package from source 
```bash
conda create -n pmnet python=3.10 -y
conda activate pmnet
pip install git+https://github.com/SeonghwanSeo/PharmacoNet.git
``` 

#### How to use the CLI

| Subcommand | One “everything included” command                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train`    | `pharmaplus train --cache-dir data/cache --out-dir runs/exp_full --val-frac 0.15 --seed 0 --device cuda:0 --epochs 10 --batch-size 64 --workers 6 --d-model 128 --retr-d 128 --lr 1e-3 --wd 1e-4 --amp --compile --log-every 20 --eval-batches 10 --aug-se3 --eval-aug-se3 --aug-trans 2.0 --pose-neg-trans 4.0 --topk 4 --edge-pairs 16 --edge-scale 5.0 --dustbin --pose-w 1.0 --xpose-w 0.3 --retr-w 1.0 --invneg-w 0.2 --temp 0.07 --max-train-steps 0 --resume runs/exp_full/best.pt --resume-opt --print-args` |
| `eval`     | `pharmaplus eval --cache-dir data/cache --ckpt runs/exp_full/best.pt --val-frac 0.15 --seed 0 --device cuda:0 --batch-size 64 --workers 6 --amp --compile --deterministic --d-model 128 --retr-d 128 --topk 4 --edge-pairs 16 --edge-scale 5.0 --dustbin --eval-aug-se3 --aug-trans 2.0 --pose-neg-trans 4.0 --pose-w 1.0 --xpose-w 0.3 --retr-w 1.0 --invneg-w 0.2 --temp 0.07 --eval-batches 10 --json`                                                                                                            |
|`visualize` | `pharmaplus visualize --ckpt runs/exp1/best.pt --item-pt data/cache/00000042.pt  --other-item-pt data/cache/00000125.pt --device cuda:0 --amp --n-confs 50 --optimize --lambda-retr 0.8  --match-topn 50 --match-thresh 0.01 --out-dir out/00000042_vs_00000125`|