python van/optim_sur_uncons.py --method fem --run_name fem-nelder --method Nelder-Mead --project optimize_nelder_2 --maxiter 1000
# for var in 1 2 3 4 5 6 7 8 9 10
for var in 1 2 3 4 5 6 7 8 9 10
do
    python van/optim_sur_uncons.py --mode data --method Nelder-Mead --model dl --d_size 3 --steps 5 --run_name scratch-data-nelder-3-5 --project optimize_nelder_2 --maxiter 1000 --dist_bound 0.00000
    python van/optim_sur_uncons.py --mode hybrid --method Nelder-Mead --model dl --d_size 3 --steps 5 --run_name scratch-hybrid-nelder-3-5 --project optimize_nelder_2 --maxiter 1000 --dist_bound 0.00000
    python van/optim_sur_uncons.py --mode data --method Nelder-Mead --model dl --d_size 3 --steps 5 --run_name data-data-nelder-3-5 --fpath van/models/data_2.h5 --project optimize_nelder_2 --maxiter 1000 --dist_bound 0
    python van/optim_sur_uncons.py --mode hybrid --method Nelder-Mead --model dl --d_size 3 --steps 5 --run_name data-hybrid-nelder-3-5 --fpath van/models/data_2.h5 --project optimize_nelder_2 --maxiter 1000 --dist_bound 0
done