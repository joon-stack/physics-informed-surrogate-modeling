for var in 1 2 3 4 5 6 7 8 9 10
do
    python burgers/run_burgers.py --mode data --fpath burgers/models/data.h5 --project burgers_1 --run_name data-data_60 --epochs 100 --task 0.01
    python burgers/run_burgers.py --mode hybrid --fpath burgers/models/data.h5 --project burgers_1 --run_name data-hybrid_60 --epochs 100 --task 0.01
    python burgers/run_burgers.py --mode data --fpath burgers/models/data.h5 --project burgers_2 --run_name data-data_60 --epochs 100 --task 0.0008
    python burgers/run_burgers.py --mode hybrid --fpath burgers/models/data.h5 --project burgers_2 --run_name data-hybrid_60 --epochs 100 --task 0.0008
    python burgers/run_burgers.py --mode data --fpath burgers/models/data.h5 --project burgers_3 --run_name data-data_60 --epochs 100 --task 0.2
    python burgers/run_burgers.py --mode hybrid --fpath burgers/models/data.h5 --project burgers_3 --run_name data-hybrid_60 --epochs 100 --task 0.2
    python burgers/run_burgers.py --mode data --fpath burgers/models/data_small.h5 --project burgers_1 --run_name data-data-small_60 --epochs 100 --task 0.01
    python burgers/run_burgers.py --mode hybrid --fpath burgers/models/data_small.h5 --project burgers_1 --run_name data-hybrid-small_60 --epochs 100 --task 0.01
    python burgers/run_burgers.py --mode data --fpath burgers/models/data_small.h5 --project burgers_2 --run_name data-data-small_60 --epochs 100 --task 0.0008
    python burgers/run_burgers.py --mode hybrid --fpath burgers/models/data_small.h5 --project burgers_2 --run_name data-hybrid-small_60 --epochs 100 --task 0.0008
    python burgers/run_burgers.py --mode data --fpath burgers/models/data_small.h5 --project burgers_3 --run_name data-data-small_60 --epochs 100 --task 0.2
    python burgers/run_burgers.py --mode hybrid --fpath burgers/models/data_small.h5 --project burgers_3 --run_name data-hybrid-small_60 --epochs 100 --task 0.2
    python burgers/run_burgers.py --mode data --project burgers_1 --run_name scratch-data_60 --epochs 100 --num_supervised_x_data 12 --num_supervised_t_data 5 --task 0.01
    python burgers/run_burgers.py --mode hybrid --project burgers_1 --run_name scratch-hybrid_60 --epochs 100 --num_supervised_x_data 12 --num_supervised_t_data 5 --task 0.01
    python burgers/run_burgers.py --mode data --project burgers_2 --run_name scratch-data_60 --epochs 100 --num_supervised_x_data 12 --num_supervised_t_data 5 --task 0.0008
    python burgers/run_burgers.py --mode hybrid --project burgers_2 --run_name scratch-hybrid_60 --epochs 100 --num_supervised_x_data 12 --num_supervised_t_data 5 --task 0.0008
    python burgers/run_burgers.py --mode data --project burgers_3 --run_name scratch-data_60 --epochs 100 --num_supervised_x_data 12 --num_supervised_t_data 5 --task 0.2
    python burgers/run_burgers.py --mode hybrid --project burgers_3 --run_name scratch-hybrid_60 --epochs 100 --num_supervised_x_data 12 --num_supervised_t_data 5 --task 0.2
done
