for var in 1 2 3 4 5 6 7 8 9 10
do
    python van/optim_surrogate.py --mode data --method trust-constr --model dl --d_size 3 --steps 10 --run_name scratch-data-trust
    python van/optim_surrogate.py --mode hybrid --method trust-constr --model dl --d_size 3 --steps 10 --run_name scratch-hybrid-trust
    python van/optim_surrogate.py --mode data --method trust-constr --model dl --d_size 3 --steps 10 --run_name data-data-trust --fpath van/models/data.h5
    python van/optim_surrogate.py --mode hybrid --method trust-constr --model dl --d_size 3 --steps 10 --run_name data-hybrid-trust --fpath van/models/data.h5
done