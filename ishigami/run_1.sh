for var in 1 2 3 4 5 6 7 8 9 10
do
    python ishigami/run_ishigami.py --mode data --fpath ishigami/models/data.h5 --project ishigami_7 --run_name data-data --epochs 100 --d_size 30 --task 28 0.3 --learning_rate 0.01
    python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/data.h5 --project ishigami_7 --run_name data-hybrid --epochs 100 --d_size 30 --task 28 0.3 --learning_rate 0.01
    python ishigami/run_ishigami.py --mode data --project ishigami_7 --run_name scratch-data --epochs 100 --d_size 80 --task 28 0.3 --learning_rate 0.01
    python ishigami/run_ishigami.py --mode hybrid --project ishigami_7 --run_name scratch-hybrid --epochs 100 --d_size 80 --task 28 0.3 --learning_rate 0.01
    python ishigami/run_ishigami.py --mode data --fpath ishigami/models/hybrid.h5 --project ishigami_7 --run_name hybrid-data --epochs 100 --d_size 30 --task 28 0.3 --learning_rate 0.01
    python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/hybrid.h5 --project ishigami_7 --run_name hybrid-hybrid --epochs 100 --d_size 30 --task 28 0.3 --learning_rate 0.01
    python ishigami/run_ishigami.py --mode data --fpath ishigami/models/data_small.h5 --project ishigami_7 --run_name data-data-small --epochs 100 --d_size 30 --task 28 0.3 --learning_rate 0.01
    python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/data_small.h5 --project ishigami_7 --run_name data-hybrid-small --epochs 100 --d_size 30 --task 28 0.3 --learning_rate 0.01
done
