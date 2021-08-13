echo --------------------fe_ing--------------------------
python src/prepare/get_features.py
echo --------------------0_demo--------------------------
python src/train/run_submit.py --k=0
echo --------------------1_demo--------------------------
python src/train/run_submit.py --k=1
echo --------------------2_demo--------------------------
python src/train/run_submit.py --k=2
echo --------------------3_demo--------------------------
python src/train/run_submit.py --k=3
echo --------------------4_demo--------------------------
python src/train/run_submit.py --k=4
echo --------------------5_demo--------------------------
python src/train/run_submit.py --k=5
echo --------------------6_demo--------------------------
python src/train/run_submit.py --k=6
echo --------------------7_demo--------------------------
python src/train/run_submit.py --k=7
echo --------------------8_demo--------------------------
python src/train/run_submit.py --k=8
echo --------------------9_demo--------------------------
python src/train/run_submit.py --k=9
