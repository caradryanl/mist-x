# Third phase: Run scripts/eval_attacks
python scripts/eval_attacks.py -m CLIPT2I --path data/outputs/celeba/clean_t2i/121 --std_path data/celeba/121 -c person
python scripts/eval_attacks.py -m CLIPT2I --path data/outputs/celeba/clean_t2i/1135 --std_path data/celeba/1135 -c person
python scripts/eval_attacks.py -m CLIPT2I --path data/outputs/celeba/clean_t2i/1422 --std_path data/celeba/1422 -c person
python scripts/eval_attacks.py -m CLIPT2I --path data/outputs/celeba/clean_t2i/1499 --std_path data/celeba/1499 -c person
python scripts/eval_attacks.py -m CLIPT2I --path data/outputs/celeba/clean_t2i/1657 --std_path data/celeba/1657 -c person

python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/clean_i2i/121 --std_path data/celeba/121 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/clean_i2i/1135 --std_path data/celeba/1135 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/clean_i2i/1422 --std_path data/celeba/1422 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/clean_i2i/1499 --std_path data/celeba/1499 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/clean_i2i/1657 --std_path data/celeba/1657 -c person

python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/clean_i2i/121 --std_path data/celeba/121 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/clean_i2i/1135 --std_path data/celeba/1135 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/clean_i2i/1422 --std_path data/celeba/1422 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/clean_i2i/1499 --std_path data/celeba/1499 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/clean_i2i/1657 --std_path data/celeba/1657 -c person

# Third phase: Run scripts/eval_attacks
python scripts/eval_attacks.py -m CLIPT2I --path data/outputs/celeba/ace_t2i/121 --std_path data/celeba/121 -c person
python scripts/eval_attacks.py -m CLIPT2I --path data/outputs/celeba/ace_t2i/1135 --std_path data/celeba/1135 -c person
python scripts/eval_attacks.py -m CLIPT2I --path data/outputs/celeba/ace_t2i/1422 --std_path data/celeba/1422 -c person
python scripts/eval_attacks.py -m CLIPT2I --path data/outputs/celeba/ace_t2i/1499 --std_path data/celeba/1499 -c person
python scripts/eval_attacks.py -m CLIPT2I --path data/outputs/celeba/ace_t2i/1657 --std_path data/celeba/1657 -c person

python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/ace_i2i/121 --std_path data/celeba/121 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/ace_i2i/1135 --std_path data/celeba/1135 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/ace_i2i/1422 --std_path data/celeba/1422 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/ace_i2i/1499 --std_path data/celeba/1499 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/ace_i2i/1657 --std_path data/celeba/1657 -c person

python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/ace_i2i/121 --std_path data/celeba/121 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/ace_i2i/1135 --std_path data/celeba/1135 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/ace_i2i/1422 --std_path data/celeba/1422 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/ace_i2i/1499 --std_path data/celeba/1499 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/ace_i2i/1657 --std_path data/celeba/1657 -c person