for var in 1 2 3 4 5 6 7 8 9 10
do
    python ishigami/run_ishigami.py --mode data --fpath ishigami/models/data.h5 --project ishigami_6 --run_name data-data --epochs 100 --d_size 30
    python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/data.h5 --project ishigami_6 --run_name data-hybrid --epochs 100 --d_size 30
    python ishigami/run_ishigami.py --mode data --project ishigami_6 --run_name scratch-data --epochs 100 --d_size 120
    python ishigami/run_ishigami.py --mode hybrid --project ishigami_6 --run_name scratch-hybrid --epochs 100 --d_size 120
done