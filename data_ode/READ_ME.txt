<Description of the dataset>

Run

    $ python3 join_1000yrs_dense_dataset.py

Then, you get the original 1000 years of dense dataset in dt=0.0005 

    q.npz
    p.npz
    dq.npz
    dp.npz

for train/eval of Plausible HNN and Baseline HNN on this heavy or dense dataset.


The 4000 years of lite dataset in dt=0.01 is
   
    q_lite.npz
    p_lite.npz
    dq_lite.npz
    dp_lite.npz

for train/eval of Plausible HNN and Block Sparse Plausible HNN on this lite dataset.
