for var in 1 2 3 4 5 6 7 8 9 10
do
    python van/run_van.py --mode data  --d_size 10 --epochs 50 --run_name scratch-data
    python van/run_van.py --mode hybrid  --d_size 10 --epochs 50 --run_name scratch-hybrid
    python van/run_van.py --mode data  --d_size 10 --epochs 50 --run_name data-data --fpath van/models/data.h5
    python van/run_van.py --mode hybrid  --d_size 10 --epochs 50 --run_name data-hybrid --fpath van/models/data.h5
done