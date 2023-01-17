for var in 1 2 3 4 5 6 7 8 9 10
do
    python burgers/test_burgers.py --mode physics --run_name data-physics-small-random --fpath ./logs/data_small.h5
    python burgers/test_burgers.py --mode hybrid --run_name data-hybrid-small-random --fpath ./logs/data_small.h5 
    python burgers/test_burgers.py --mode data --run_name hybrid-data-small-random --fpath ./logs/hybrid_small.h5  
    python burgers/test_burgers.py --mode physics --run_name hybrid-physics-small-random --fpath ./logs/hybrid_small.h5   
done