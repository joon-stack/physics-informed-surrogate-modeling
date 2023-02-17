# python van/optim_surrogate.py --method fem --run_name fem-slsqp --method SLSQP --project optimize_slsqp --maxiter 200
for var in 1
# for var in 1 2 3 4 5 6 7 8 9 10
do
    python van/optim_surrogate.py --mode hybrid --method SLSQP --model dl --d_size 100 --steps 50 --run_name data-hybrid-SLSQP --fpath van/models/data.h5 --project optimize_slsqp --maxiter 200 --dist_bound 0.005
    # python van/optim_surrogate.py --mode data --method SLSQP --model dl --d_size 10 --steps 50 --run_name data-data-SLSQP --fpath van/models/data.h5 --project optimize_slsqp --maxiter 200 --dist_bound 0.01
    # python van/optim_surrogate.py --mode data --method SLSQP --model dl --d_size 10 --steps 25 --run_name scratch-data-SLSQP
    # python van/optim_surrogate.py --mode hybrid --method SLSQP --model dl --d_size 10 --steps 25 --run_name scratch-hybrid-SLSQP
done