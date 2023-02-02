for var in 1 2 3 4 5 6 7 8 9 10
do
    python tube/run_tube.py --project tube_1 --mode hybrid --task_out 0 --seed 1234 --d_size 20 --run_name data-hybrid-20-50 --fpath tube/models/data-50.h5 --device_no 0
    python tube/run_tube.py --project tube_1 --mode data --task_out 0 --seed 1234 --d_size 20 --run_name data-data-20-50 --fpath tube/models/data-50.h5 --device_no 0
    python tube/run_tube.py --project tube_2 --mode hybrid --task_out 1 --seed 1234 --d_size 20 --run_name data-hybrid-20-50 --fpath tube/models/data-50.h5 --device_no 0
    python tube/run_tube.py --project tube_2 --mode data --task_out 1 --seed 1234 --d_size 20 --run_name data-data-20-50 --fpath tube/models/data-50.h5 --device_no 0
    python tube/run_tube.py --project tube_3 --mode hybrid --task_out 2 --seed 1234 --d_size 20 --run_name data-hybrid-20-50 --fpath tube/models/data-50.h5 --device_no 0
    python tube/run_tube.py --project tube_3 --mode data --task_out 2 --seed 1234 --d_size 20 --run_name data-data-20-50 --fpath tube/models/data-50.h5 --device_no 0
    python tube/run_tube.py --project tube_1 --mode hybrid --task_out 0 --seed 1234 --d_size 70 --run_name hybrid-70
    python tube/run_tube.py --project tube_1 --mode data --task_out 0 --seed 1234 --d_size 70 --run_name data-70
    python tube/run_tube.py --project tube_2 --mode hybrid --task_out 1 --seed 1234 --d_size 70 --run_name hybrid-70
    python tube/run_tube.py --project tube_2 --mode data --task_out 1 --seed 1234 --d_size 70 --run_name data-70
    python tube/run_tube.py --project tube_3 --mode hybrid --task_out 2 --seed 1234 --d_size 70 --run_name hybrid-70
    python tube/run_tube.py --project tube_3 --mode data --task_out 2 --seed 1234 --d_size 70 --run_name data-70
done
