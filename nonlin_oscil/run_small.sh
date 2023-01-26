for var in 1 2 3 4 5 6 7 8 9 10
do
    # python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 1.0 1.0 0.5 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name scratch-hybrid-90-small --project oscilator_1 --device_no 1
    python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 1.0 1.0 0.5 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name data-hybrid-290-small --fpath nonlin_oscil/models/data_small.h5 --project oscilator_1 --device_no 1
    # python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 1.0 1.0 0.5 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name scratch-data-290-small --project oscilator_1 --device_no 1
    python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 1.0 1.0 0.5 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name data-data-290-small --fpath nonlin_oscil/models/data_small.h5 --project oscilator_1 --device_no 1
    # python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 1.0 0.8 0.7 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name scratch-hybrid-290-small --project oscilator_2 --device_no 1
    python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 1.0 0.8 0.7 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name data-hybrid-290-small --fpath nonlin_oscil/models/data_small.h5 --project oscilator_2 --device_no 1
    # python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 1.0 0.8 0.7 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name scratch-data-290-small --project oscilator_2 --device_no 1
    python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 1.0 0.8 0.7 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name data-data-290-small --fpath nonlin_oscil/models/data_small.h5 --project oscilator_2 --device_no 1
    # python nonlin_oscil/run_nonlin_oscil.py --task 0.5 0.1 1.0 1.0 0.7 0.8 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name scratch-hybrid-290-small --project oscilator_3 --device_no 1
    python nonlin_oscil/run_nonlin_oscil.py --task 0.5 0.1 1.0 1.0 0.7 0.8 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name data-hybrid-290-small --fpath nonlin_oscil/models/data_small.h5 --project oscilator_3 --device_no 1
    # python nonlin_oscil/run_nonlin_oscil.py --task 0.5 0.1 1.0 1.0 0.7 0.8 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name scratch-data-290-small --project oscilator_3 --device_no 1
    python nonlin_oscil/run_nonlin_oscil.py --task 0.5 0.1 1.0 1.0 0.7 0.8 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name data-data-290-small --fpath nonlin_oscil/models/data_small.h5 --project oscilator_3 --device_no 1
    # python nonlin_oscil/run_nonlin_oscil.py --task 1.5 0.3 1.3 0.9 1.5 0.5 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name scratch-hybrid-290-small --project oscilator_4 --device_no 1
    python nonlin_oscil/run_nonlin_oscil.py --task 1.5 0.3 1.3 0.9 1.5 0.5 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name data-hybrid-290-small --fpath nonlin_oscil/models/data_small.h5 --project oscilator_4 --device_no 1
    # python nonlin_oscil/run_nonlin_oscil.py --task 1.5 0.3 1.3 0.9 1.5 0.5 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name scratch-data-290-small --project oscilator_4 --device_no 1
    python nonlin_oscil/run_nonlin_oscil.py --task 1.5 0.3 1.3 0.9 1.5 0.5 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name data-data-290-small --fpath nonlin_oscil/models/data_small.h5 --project oscilator_4 --device_no 1
    # python nonlin_oscil/run_nonlin_oscil.py --task 0.6 0.05 1.0 0.7 0.9 0.3 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name scratch-hybrid-290-small --project oscilator_5 --device_no 1
    python nonlin_oscil/run_nonlin_oscil.py --task 0.6 0.05 1.0 0.7 0.9 0.3 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name data-hybrid-290-small --fpath nonlin_oscil/models/data_small.h5 --project oscilator_5 --device_no 1
    # python nonlin_oscil/run_nonlin_oscil.py --task 0.6 0.05 1.0 0.7 0.9 0.3 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name scratch-data-290-small --project oscilator_5 --device_no 1
    python nonlin_oscil/run_nonlin_oscil.py --task 0.6 0.05 1.0 0.7 0.9 0.3 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name data-data-290-small --fpath nonlin_oscil/models/data_small.h5 --project oscilator_5 --device_no 1
    # python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 0.7 1.1 0.3 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name scratch-hybrid-290-small --project oscilator_6 --device_no 1
    python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 0.7 1.1 0.3 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name data-hybrid-290-small --fpath nonlin_oscil/models/data_small.h5 --project oscilator_6 --device_no 1
    # python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 0.7 1.1 0.3 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name scratch-data-290-small --project oscilator_6 --device_no 1
    python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 0.7 1.1 0.3 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name data-data-290-small --fpath nonlin_oscil/models/data_small.h5 --project oscilator_6 --device_no 1
done