for var in 1 2 3 4 5 6 7 8 9 10
do
    python ishigami/run_ishigami.py --mode data --fpath ishigami/models/data.h5 --project ishigami_1 --run_name data-data_60 --epochs 100 --d_size 60 --task 7 0.1
    python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/data.h5 --project ishigami_1 --run_name data-hybrid_60 --epochs 100 --d_size 60 --task 7 0.1
    python ishigami/run_ishigami.py --mode data --fpath ishigami/models/data.h5 --project ishigami_2 --run_name data-data_60 --epochs 100 --d_size 60 --task 21 0.3
    python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/data.h5 --project ishigami_2 --run_name data-hybrid_60 --epochs 100 --d_size 60 --task 21 0.3
    python ishigami/run_ishigami.py --mode data --fpath ishigami/models/data.h5 --project ishigami_3 --run_name data-data_60 --epochs 100 --d_size 60 --task 42 0.1
    python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/data.h5 --project ishigami_3 --run_name data-hybrid_60 --epochs 100 --d_size 60 --task 42 0.1
    python ishigami/run_ishigami.py --mode data --fpath ishigami/models/data.h5 --project ishigami_4 --run_name data-data_60 --epochs 100 --d_size 60 --task 21 0.6
    python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/data.h5 --project ishigami_4 --run_name data-hybrid_60 --epochs 100 --d_size 60 --task 21 0.6
    python ishigami/run_ishigami.py --mode data --fpath ishigami/models/data.h5 --project ishigami_5 --run_name data-data_60 --epochs 100 --d_size 60 --task 7 0.03
    python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/data.h5 --project ishigami_5 --run_name data-hybrid_60 --epochs 100 --d_size 60 --task 7 0.03
    python ishigami/run_ishigami.py --mode data --fpath ishigami/models/data.h5 --project ishigami_6 --run_name data-data_60 --epochs 100 --d_size 60 --task 21 0.1
    python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/data.h5 --project ishigami_6 --run_name data-hybrid_60 --epochs 100 --d_size 60 --task 21 0.1
    python ishigami/run_ishigami.py --mode data --fpath ishigami/models/data_small.h5 --project ishigami_1 --run_name data-data-small_60 --epochs 100 --d_size 60 --task 7 0.1
    python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/data_small.h5 --project ishigami_1 --run_name data-hybrid-small_60 --epochs 100 --d_size 60 --task 7 0.1
    python ishigami/run_ishigami.py --mode data --fpath ishigami/models/data_small.h5 --project ishigami_2 --run_name data-data-small_60 --epochs 100 --d_size 60 --task 21 0.3
    python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/data_small.h5 --project ishigami_2 --run_name data-hybrid-small_60 --epochs 100 --d_size 60 --task 21 0.3
    python ishigami/run_ishigami.py --mode data --fpath ishigami/models/data_small.h5 --project ishigami_3 --run_name data-data-small_60 --epochs 100 --d_size 60 --task 42 0.1
    python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/data_small.h5 --project ishigami_3 --run_name data-hybrid-small_60 --epochs 100 --d_size 60 --task 42 0.1
    python ishigami/run_ishigami.py --mode data --fpath ishigami/models/data_small.h5 --project ishigami_4 --run_name data-data-small_60 --epochs 100 --d_size 60 --task 21 0.6
    python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/data_small.h5 --project ishigami_4 --run_name data-hybrid-small_60 --epochs 100 --d_size 60 --task 21 0.6
    python ishigami/run_ishigami.py --mode data --fpath ishigami/models/data_small.h5 --project ishigami_5 --run_name data-data-small_60 --epochs 100 --d_size 60 --task 7 0.03
    python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/data_small.h5 --project ishigami_5 --run_name data-hybrid-small_60 --epochs 100 --d_size 60 --task 7 0.03
    python ishigami/run_ishigami.py --mode data --fpath ishigami/models/data_small.h5 --project ishigami_6 --run_name data-data-small_60 --epochs 100 --d_size 60 --task 21 0.1    
    python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/data_small.h5 --project ishigami_6 --run_name data-hybrid-small_60 --epochs 100 --d_size 60 --task 21 0.1
    python ishigami/run_ishigami.py --mode data --project ishigami_1 --run_name scratch-data_60 --epochs 100 --d_size 110 --task 7 0.1
    python ishigami/run_ishigami.py --mode hybrid --project ishigami_1 --run_name scratch-hybrid_60 --epochs 100 --d_size 110 --task 7 0.1
    python ishigami/run_ishigami.py --mode data --project ishigami_2 --run_name scratch-data_60 --epochs 100 --d_size 110 --task 21 0.3
    python ishigami/run_ishigami.py --mode hybrid --project ishigami_2 --run_name scratch-hybrid_60 --epochs 100 --d_size 110 --task 21 0.3
    python ishigami/run_ishigami.py --mode data --project ishigami_3 --run_name scratch-data_60 --epochs 100 --d_size 110 --task 42 0.1
    python ishigami/run_ishigami.py --mode hybrid --project ishigami_3 --run_name scratch-hybrid_60 --epochs 100 --d_size 110 --task 42 0.1
    python ishigami/run_ishigami.py --mode data --project ishigami_4 --run_name scratch-data_60 --epochs 100 --d_size 110 --task 21 0.6
    python ishigami/run_ishigami.py --mode hybrid --project ishigami_4 --run_name scratch-hybrid_60 --epochs 100 --d_size 110 --task 21 0.6
    python ishigami/run_ishigami.py --mode data --project ishigami_5 --run_name scratch-data_60 --epochs 100 --d_size 110 --task 7 0.03
    python ishigami/run_ishigami.py --mode hybrid --project ishigami_5 --run_name scratch-hybrid_60 --epochs 100 --d_size 110 --task 7 0.03
    python ishigami/run_ishigami.py --mode data --project ishigami_6 --run_name scratch-data_60 --epochs 100 --d_size 110 --task 21 0.1
    python ishigami/run_ishigami.py --mode hybrid --project ishigami_6 --run_name scratch-hybrid_60 --epochs 100 --d_size 110 --task 21 0.1
    # python ishigami/run_ishigami.py --mode data --fpath ishigami/models/hybrid.h5 --project ishigami_1 --run_name hybrid-data_60 --epochs 100 --d_size 60 --task 7 0.1
    # python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/hybrid.h5 --project ishigami_1 --run_name hybrid-hybrid_60 --epochs 100 --d_size 60 --task 7 0.1
    # python ishigami/run_ishigami.py --mode data --fpath ishigami/models/hybrid.h5 --project ishigami_2 --run_name hybrid-data_60 --epochs 100 --d_size 60 --task 21 0.3
    # python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/hybrid.h5 --project ishigami_2 --run_name hybrid-hybrid_60 --epochs 100 --d_size 60 --task 21 0.3
    # python ishigami/run_ishigami.py --mode data --fpath ishigami/models/hybrid.h5 --project ishigami_3 --run_name hybrid-data_60 --epochs 100 --d_size 60 --task 42 0.1
    # python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/hybrid.h5 --project ishigami_3 --run_name hybrid-hybrid_60 --epochs 100 --d_size 60 --task 42 0.1
    # python ishigami/run_ishigami.py --mode data --fpath ishigami/models/hybrid.h5 --project ishigami_4 --run_name hybrid-data_60 --epochs 100 --d_size 60 --task 21 0.6
    # python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/hybrid.h5 --project ishigami_4 --run_name hybrid-hybrid_60 --epochs 100 --d_size 60 --task 21 0.6
    # python ishigami/run_ishigami.py --mode data --fpath ishigami/models/hybrid.h5 --project ishigami_5 --run_name hybrid-data_60 --epochs 100 --d_size 60 --task 7 0.03
    # python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/hybrid.h5 --project ishigami_5 --run_name hybrid-hybrid_60 --epochs 100 --d_size 60 --task 7 0.03
    # python ishigami/run_ishigami.py --mode data --fpath ishigami/models/hybrid.h5 --project ishigami_6 --run_name hybrid-data_60 --epochs 100 --d_size 60 --task 21 0.1
    # python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/hybrid.h5 --project ishigami_6 --run_name hybrid-hybrid_60 --epochs 100 --d_size 60 --task 21 0.1
    python ishigami/run_ishigami.py --mode data --fpath ishigami/models/data.h5 --project ishigami_7 --run_name data-data_60 --epochs 100 --d_size 60 --task 28 0.3 --learning_rate 0.01
    python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/data.h5 --project ishigami_7 --run_name data-hybrid_60 --epochs 100 --d_size 60 --task 28 0.3 --learning_rate 0.01
    python ishigami/run_ishigami.py --mode data --project ishigami_7 --run_name scratch-data_60 --epochs 100 --d_size 110 --task 28 0.3 --learning_rate 0.01
    python ishigami/run_ishigami.py --mode hybrid --project ishigami_7 --run_name scratch-hybrid_60 --epochs 100 --d_size 110 --task 28 0.3 --learning_rate 0.01
    # python ishigami/run_ishigami.py --mode data --fpath ishigami/models/hybrid.h5 --project ishigami_7 --run_name hybrid-data --epochs 100 --d_size 60 --task 28 0.3 --learning_rate 0.01
    # python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/hybrid.h5 --project ishigami_7 --run_name hybrid-hybrid --epochs 100 --d_size 60 --task 28 0.3 --learning_rate 0.01
    python ishigami/run_ishigami.py --mode data --fpath ishigami/models/data_small.h5 --project ishigami_7 --run_name data-data-small_60 --epochs 100 --d_size 60 --task 28 0.3 --learning_rate 0.01
    python ishigami/run_ishigami.py --mode hybrid --fpath ishigami/models/data_small.h5 --project ishigami_7 --run_name data-hybrid-small_60 --epochs 100 --d_size 60 --task 28 0.3 --learning_rate 0.01
done
