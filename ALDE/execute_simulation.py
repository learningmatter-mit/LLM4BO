from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch
import random
import os, time
import multiprocessing as mp
import warnings
from src.optimize import BayesianOptimization, BO_ARGS
import src.objectives as objectives
import src.utils as utils
'''
Script to repdouce all of the active learning simulations on GB1 and TrpB datasets. Launches optimization runs as separate processes.
'''



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--names", nargs="+", default=["GB1", "TrpB"])
    parser.add_argument("--encodings", nargs="+", default=["onehot"])
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--n_pseudorand_init", type=int, default=96)
    parser.add_argument("--budget", type=int, default=384)
    parser.add_argument("--output_path", type=str, default='results/96_384_simulations/')
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--seed_index", type=int, default=0)
    parser.add_argument("--kernel", type=str, default="RBF", choices=["RBF"])
    parser.add_argument("--xi", type=float, default=4, help="trade-off parameter for the UCB and EI acquisition function. Using EI with xi > 1 will default to xi = 0.0")
    parser.add_argument("--activation", type=str, default="lrelu")
    parser.add_argument("--min_noise", type=float, default=1e-6)
    parser.add_argument("--train_iter", type=int, default=300)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--acq_fn", nargs="+", default=['GREEDY', 'UCB', 'TS', 'EI', 'EPSILON_GREEDY'])
    parser.add_argument("--model_type", type=str, default="DNN_ENSEMBLE", choices=["DNN_ENSEMBLE", "GP_BOTORCH", "DKL_BOTORCH", "BOOSTING_ENSEMBLE"])
    parser.add_argument("--start_from", type=str, default=None, help="Path to folder containing .pt files with starting indices")

    args = parser.parse_args()
    print(args)
    warnings.filterwarnings("ignore")

    #8 different objectives (2 datasets, each with 4 encodings)
    for protein in args.names:
        for encoding in args.encodings:
            device = args.device
            print(device)

            obj = objectives.Combo(protein, encoding)

            obj_fn = obj.objective
            domain = obj.get_domain()
            ymax = obj.get_max()
            disc_X = obj.get_points()[0]
            disc_y = obj.get_points()[1]
            batch_size = args.batch_size #number of samples to query in each round of active learning

            n_pseudorand_init = args.n_pseudorand_init #number of initial random samples
            budget = args.budget #total number of samples to query, not including random initializations

            try:
                mp.set_start_method('spawn')
            except:
                print('Context already set.')
            
            # make dir to hold tensors
            path = args.output_path
            subdir = path + '/' + protein + '/' + encoding + '/'
            os.makedirs(subdir, exist_ok=True)
            os.system('cp ' + __file__ + ' ' + subdir) #save the script that generated the results
            print('Script stored.')

            runs = args.runs #number of times to repeat the simulation
            index = args.seed_index #index of the first run (reads from rndseed.txt to choose the seed)
            seeds = []

            with open('src/rndseed.txt', 'r') as f:
                lines = f.readlines()
                for i in range(runs):
                    print('run index: {}'.format(index+i))
                    line = lines[i+index].strip('\n')
                    print('seed: {}'.format(int(line)))
                    seeds.append(int(line))
            
            # Load starting indices from files if start_from is specified
            start_indices_files = []
            if args.start_from:
                start_from_dir = os.path.join(args.start_from, protein, encoding)
                if os.path.exists(start_from_dir):
                    pt_files = [f for f in os.listdir(start_from_dir) if f.endswith('.pt')]
                    pt_files.sort()  # Sort to ensure consistent ordering
                    start_indices_files = [os.path.join(start_from_dir, f) for f in pt_files]
                    print(f'Found {len(start_indices_files)} .pt files in {start_from_dir}')
                else:
                    print(f'Warning: start_from directory {start_from_dir} does not exist, falling back to random initialization')
                    start_indices_files = []
            
            # Only set CUDNN settings if using CUDA device
            if device == "cuda" and torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            
            arg_list = []
            start_time = time.time()
            #loop through each of the indices
            for r in range(index, index + runs):
                seed = seeds[r - index]
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                
                # Only set CUDA seeds if using CUDA device
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                print('Cuda available: {}'.format(torch.cuda.is_available()))

                # Generate or load starting indices
                if args.start_from and start_indices_files:
                    # Use indices from file for this run
                    run_file_index = (r - index) % len(start_indices_files)  # Cycle through files if fewer files than runs
                    indices_file = start_indices_files[run_file_index]
                    print(f'Loading starting indices from {indices_file}')
                    
                    try:
                        loaded_indices = torch.load(indices_file, map_location='cpu')
                        # Take first n_pseudorand_init indices
                        if len(loaded_indices) >= n_pseudorand_init:
                            start_indices = loaded_indices[:n_pseudorand_init]
                        else:
                            print(f'Warning: File {indices_file} has only {len(loaded_indices)} indices, need {n_pseudorand_init}')
                            start_indices = loaded_indices
                        
                        # Get corresponding x and y values
                        start_x = disc_X[start_indices]
                        start_y = disc_y[start_indices]
                        print(f'Loaded {len(start_indices)} starting indices from file')
                    except Exception as e:
                        print(f'Error loading indices from {indices_file}: {e}')
                        print('Falling back to random initialization')
                        start_x, start_y, start_indices = utils.samp_discrete(n_pseudorand_init, obj, seed)
                else:
                    # Use random initialization (original behavior)
                    start_x, start_y, start_indices = utils.samp_discrete(n_pseudorand_init, obj, seed)
                if budget != 0:
                    _, randy, rand_indices = utils.samp_discrete(budget, obj, seed)
                    rand_indices = torch.cat((start_indices, rand_indices), 0)
                else:
                    rand_indices = start_indices
                
                # temp = []
                # for n in range(budget + 96 + 1):
                #     m = torch.max(randy[:n + n_pseudorand_init])
                #     reg = torch.reshape(torch.abs(ymax - m), (1, -1))
                #     temp.append(reg)
                # tc = torch.cat(temp, 0)
                # tc = torch.reshape(tc, (1, -1))
                # torch.save(tc, subdir + 'Random_' + str(r + 1) + 'regret.pt')
                # torch.save(randy, subdir + 'Random_' + str(r + 1) + 'y.pt')


                torch.save(rand_indices, subdir + 'Random_' + str(r + 1) + 'indices.pt')
                print('Random search done.')


                kernel=args.kernel #kernel must be radial basis function, only applies to GP_BOTORCH and DKL_BOTORCH
                for mtype in [args.model_type]: # "BOOSTING_ENSEMBLE", "", "DKL_BOTORCH", DNN_ENSEMBLE
                    for acq_fn in args.acq_fn: # 'GREEDY', 'UCB', 'TS', 'AGENT', 
                        
                        dropout=args.dropout #dropout rate, only applies to neural networks models (DNN_ENSEMBLE and DKL_BOTORCH)

                        if mtype == 'GP_BOTORCH' and 'ESM2' in encoding:
                            lr = 1e-1
                        else:
                            lr = 1e-3
                        
                        num_simult_jobs = 1 #number of simulations to run in parallel

                        #set the architecture of the neural network
                        if 'DNN' in mtype and 'ENSEMBLE' in mtype:
                            if 'onehot' in encoding:
                                arc  = [domain[0].size(-1), 30, 30, 1]
                            elif 'AA' in encoding:
                                arc  = [domain[0].size(-1), 8, 8, 1]
                            elif 'georgiev' in encoding:
                                arc  = [domain[0].size(-1), 30, 30, 1]
                            elif 'ESM2' in encoding:
                                arc  = [domain[0].size(-1), 500, 150, 50, 1] 
                        elif 'GP' in mtype:
                            arc = [domain[0].size(-1), 1]
                        elif 'DKL' in mtype:
                            if 'onehot' in encoding:
                                arc  = [domain[0].size(-1), 30, 30, 1]
                            elif 'AA' in encoding:
                                arc  = [domain[0].size(-1), 8, 8, 1]
                            elif 'georgiev' in encoding:
                                arc  = [domain[0].size(-1), 30, 30, 1]
                            else:
                                arc  = [domain[0].size(-1), 500, 150, 50, 1]
                        else:
                            arc = [domain[0].size(-1), 1]

                        #filename
                        fname = mtype + '-DO-' + str(dropout) + '-' + kernel + '-' + acq_fn + '-' + str(arc[-2:]) + '_' + str(r + 1)

                        sim_args = BO_ARGS(
                            mtype=mtype,
                            kernel=kernel,
                            acq_fn=acq_fn,
                            xi=args.xi, #xi term, only applies to UCB
                            architecture=arc,
                            activation=args.activation,
                            min_noise=args.min_noise,
                            trainlr=lr,
                            train_iter=args.train_iter,
                            dropout=dropout,
                            mcdropout=0,
                            verbose=args.verbose,
                            bb_fn=obj_fn,
                            bb_obj=obj,
                            domain=domain,
                            disc_X=disc_X,
                            disc_y=disc_y,
                            noise_std=0,
                            n_rand_init=0, #additional random inits
                            budget=budget,
                            query_cost=1,
                            queries_x=start_x,
                            queries_y=start_y,
                            indices=start_indices,
                            savedir=subdir+fname,
                            batch_size = batch_size,
                            device=device
                        )
                        arg_list.append((sim_args, seed))

            with mp.Pool(num_simult_jobs) as pool:
                pool.starmap(BayesianOptimization.run, arg_list)
                pool.close()
                pool.join()
                print(f'Total runtime: {time.time()-start_time}')
            with open("timeit.txt", "a") as f:
                f.write(f"{protein},{mtype},{kernel},{acq_fn},{arc[-2:]},{budget},{time.time()-start_time:.2f}\n")
            print('Tensors will be saved in {}'.format(subdir))
