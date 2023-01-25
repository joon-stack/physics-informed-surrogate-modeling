for var in 1 2 3 4 5 6 7 8 9 10
do
    python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 1.0 1.0 0.5 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name scratch-hybrid-40 --project oscilator_1
    python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 1.0 1.0 0.5 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name data-hybrid-40 --fpath nonlin_oscil/models/data.h5 --project oscilator_1
    python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 1.0 1.0 0.5 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name scratch-data-40 --project oscilator_1
    python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 1.0 1.0 0.5 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name data-data-40 --fpath nonlin_oscil/models/data.h5 --project oscilator_1 
    python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 1.0 0.8 0.7 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name scratch-hybrid-40 --project oscilator_2
    python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 1.0 0.8 0.7 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name data-hybrid-40 --fpath nonlin_oscil/models/data.h5 --project oscilator_2
    python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 1.0 0.8 0.7 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name scratch-data-40 --project oscilator_2
    python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 1.0 0.8 0.7 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name data-data-40 --fpath nonlin_oscil/models/data.h5 --project oscilator_2 
    python nonlin_oscil/run_nonlin_oscil.py --task 0.5 0.1 1.0 1.0 0.7 0.8 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name scratch-hybrid-40 --project oscilator_3
    python nonlin_oscil/run_nonlin_oscil.py --task 0.5 0.1 1.0 1.0 0.7 0.8 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name data-hybrid-40 --fpath nonlin_oscil/models/data.h5 --project oscilator_3
    python nonlin_oscil/run_nonlin_oscil.py --task 0.5 0.1 1.0 1.0 0.7 0.8 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name scratch-data-40 --project oscilator_3
    python nonlin_oscil/run_nonlin_oscil.py --task 0.5 0.1 1.0 1.0 0.7 0.8 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name data-data-40 --fpath nonlin_oscil/models/data.h5 --project oscilator_3 
    python nonlin_oscil/run_nonlin_oscil.py --task 1.5 0.3 1.3 0.9 1.5 0.5 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name scratch-hybrid-40 --project oscilator_4
    python nonlin_oscil/run_nonlin_oscil.py --task 1.5 0.3 1.3 0.9 1.5 0.5 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name data-hybrid-40 --fpath nonlin_oscil/models/data.h5 --project oscilator_4
    python nonlin_oscil/run_nonlin_oscil.py --task 1.5 0.3 1.3 0.9 1.5 0.5 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name scratch-data-40 --project oscilator_4
    python nonlin_oscil/run_nonlin_oscil.py --task 1.5 0.3 1.3 0.9 1.5 0.5 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name data-data-40 --fpath nonlin_oscil/models/data.h5 --project oscilator_4 
    python nonlin_oscil/run_nonlin_oscil.py --task 0.6 0.05 1.0 0.7 0.9 0.3 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name scratch-hybrid-40 --project oscilator_5
    python nonlin_oscil/run_nonlin_oscil.py --task 0.6 0.05 1.0 0.7 0.9 0.3 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name data-hybrid-40 --fpath nonlin_oscil/models/data.h5 --project oscilator_5
    python nonlin_oscil/run_nonlin_oscil.py --task 0.6 0.05 1.0 0.7 0.9 0.3 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name scratch-data-40 --project oscilator_5
    python nonlin_oscil/run_nonlin_oscil.py --task 0.6 0.05 1.0 0.7 0.9 0.3 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name data-data-40 --fpath nonlin_oscil/models/data.h5 --project oscilator_5 
    python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 0.7 1.1 0.3 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name scratch-hybrid-40 --project oscilator_6
    python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 0.7 1.1 0.3 --d_size 40 --f_size 100 --mode hybrid --epochs 1000 --run_name data-hybrid-40 --fpath nonlin_oscil/models/data.h5 --project oscilator_6
    python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 0.7 1.1 0.3 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name scratch-data-40 --project oscilator_6
    python nonlin_oscil/run_nonlin_oscil.py --task 1.0 0.1 1.0 0.7 1.1 0.3 --d_size 40 --f_size 100 --mode data --epochs 1000 --run_name data-data-40 --fpath nonlin_oscil/models/data.h5 --project oscilator_6  
done