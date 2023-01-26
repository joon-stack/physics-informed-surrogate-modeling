for var in 1 2 3 4 5 6 7 8 9 10
do
    python highdim/run_highdim.py --project highdim_1 --run_name scratch-data-20 --mode data --d_size 20 --seed 1004 --task_out 0
    python highdim/run_highdim.py --project highdim_1 --run_name scratch-hybrid-20 --mode hybrid --d_size 20 --seed 1004 --task_out 0
    python highdim/run_highdim.py --project highdim_1 --run_name scratch-data-120 --mode data --d_size 120 --seed 1004 --task_out 0
    python highdim/run_highdim.py --project highdim_1 --run_name scratch-hybrid-120 --mode hybrid --d_size 120 --seed 1004 --task_out 0
    python highdim/run_highdim.py --project highdim_1 --run_name data-data-20 --mode data --d_size 20 --fpath highdim/models/data_40dim.h5 --seed 1004 --task_out 0
    python highdim/run_highdim.py --project highdim_1 --run_name data-hybrid-20 --mode hybrid --d_size 20 --fpath highdim/models/data_40dim.h5 --seed 1004 --task_out 0
    python highdim/run_highdim.py --project highdim_1 --run_name data-data-20-small --mode data --d_size 20 --fpath highdim/models/data_40dim_small.h5 --seed 1004 --task_out 0
    python highdim/run_highdim.py --project highdim_1 --run_name data-hybrid-20-small --mode hybrid --d_size 20 --fpath highdim/models/data_40dim_small.h5 --seed 1004 --task_out 0
    python highdim/run_highdim.py --project highdim_2 --run_name scratch-data-20 --mode data --d_size 20 --seed 1004 --task_out 1
    python highdim/run_highdim.py --project highdim_2 --run_name scratch-hybrid-20 --mode hybrid --d_size 20 --seed 1004 --task_out 1
    python highdim/run_highdim.py --project highdim_2 --run_name scratch-data-120 --mode data --d_size 120 --seed 1004 --task_out 1
    python highdim/run_highdim.py --project highdim_2 --run_name scratch-hybrid-120 --mode hybrid --d_size 120 --seed 1004 --task_out 1
    python highdim/run_highdim.py --project highdim_2 --run_name data-data-20 --mode data --d_size 20 --fpath highdim/models/data_40dim.h5 --seed 1004 --task_out 1
    python highdim/run_highdim.py --project highdim_2 --run_name data-hybrid-20 --mode hybrid --d_size 20 --fpath highdim/models/data_40dim.h5 --seed 1004 --task_out 1
    python highdim/run_highdim.py --project highdim_2 --run_name data-data-20-small --mode data --d_size 20 --fpath highdim/models/data_40dim_small.h5 --seed 1004 --task_out 1
    python highdim/run_highdim.py --project highdim_2 --run_name data-hybrid-20-small --mode hybrid --d_size 20 --fpath highdim/models/data_40dim_small.h5 --seed 1004 --task_out 1
    python highdim/run_highdim.py --project highdim_3 --run_name scratch-data-20 --mode data --d_size 20 --seed 1004 --task_out 2
    python highdim/run_highdim.py --project highdim_3 --run_name scratch-hybrid-20 --mode hybrid --d_size 20 --seed 1004 --task_out 2
    python highdim/run_highdim.py --project highdim_3 --run_name scratch-data-120 --mode data --d_size 120 --seed 1004 --task_out 2
    python highdim/run_highdim.py --project highdim_3 --run_name scratch-hybrid-120 --mode hybrid --d_size 120 --seed 1004 --task_out 2
    python highdim/run_highdim.py --project highdim_3 --run_name data-data-20 --mode data --d_size 20 --fpath highdim/models/data_40dim.h5 --seed 1004 --task_out 2
    python highdim/run_highdim.py --project highdim_3 --run_name data-hybrid-20 --mode hybrid --d_size 20 --fpath highdim/models/data_40dim.h5 --seed 1004 --task_out 2
    python highdim/run_highdim.py --project highdim_3 --run_name data-data-20-small --mode data --d_size 20 --fpath highdim/models/data_40dim_small.h5 --seed 1004 --task_out 2
    python highdim/run_highdim.py --project highdim_3 --run_name data-hybrid-20-small --mode hybrid --d_size 20 --fpath highdim/models/data_40dim_small.h5 --seed 1004 --task_out 2
done
