import math
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils import load_model, move_to
from utils.data_utils import save_dataset
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from utils.functions import parse_softmax_temperature
from nets.attention_model import set_decode_type
from setuptools.dist import sequence


mp = torch.multiprocessing.get_context('spawn')


def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]


def eval_dataset_mp(args):
    (dataset_path, width, softmax_temp, opts, i, num_processes) = args

    model, _ = load_model(opts.model)
    val_size = opts.val_size // num_processes
    dataset = model.problem.make_dataset(filename=dataset_path, num_samples=val_size, offset=opts.offset + val_size * i)
    device = torch.device("cuda:{}".format(i))

    return _eval_dataset(model, dataset, width, softmax_temp, opts, device)


def eval_dataset(dataset_path, width, softmax_temp, opts):
    # Even with multiprocessing, we load the model here since it contains the name where to write results
    seeder, _ = load_model(opts.seeder)
    

    if opts.problem == 'tsp':
        reviser, _ = load_model(opts.reviser,is_local=True)

        if opts.reviser_2 is not None:
            reviser_2, _ = load_model(opts.reviser_2,is_local=True)
        else:
            reviser_2 = None
    else:
        reviser = None
        reviser_2 = None

    use_cuda = torch.cuda.is_available() and not opts.no_cuda

    if opts.multiprocessing:
        assert use_cuda, "Can only do multiprocessing with cuda"
        num_processes = torch.cuda.device_count()
        assert opts.val_size % num_processes == 0

        with mp.Pool(num_processes) as pool:
            results = list(itertools.chain.from_iterable(pool.map(
                eval_dataset_mp,
                [(dataset_path, width, softmax_temp, opts, i, num_processes) for i in range(num_processes)]
            )))

    else:
        device = torch.device("cuda:0" if use_cuda else "cpu")
        dataset = seeder.problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=opts.offset)
        results = _eval_dataset(seeder, dataset, width, softmax_temp, opts, device,reviser=reviser,reviser_2=reviser_2)

    # This is parallelism, even if we use multiprocessing (we report as if we did not use multiprocessing, e.g. 1 GPU)
    parallelism = opts.eval_batch_size

    costs, costs_revised,tours, durations = zip(*results)  # Not really costs since they should be negative

    print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
    if opts.problem =="tsp":

        print("Average cost_revised: {} +- {}".format(np.mean(costs_revised), 2 * np.std(costs_revised) / np.sqrt(len(costs_revised))))
    print("Average serial duration: {} +- {}".format(
        np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
    print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
    print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))



    return costs, tours, durations


def _eval_dataset(model, dataset, width, softmax_temp, opts, device,reviser=None,reviser_2=None):

    model.to(device)
    model.eval()
    if reviser is not None:
        reviser.to(device)
        reviser.eval()
        reviser.set_decode_type("greedy")
    if reviser_2 is not None:
        reviser_2.to(device)
        reviser_2.eval()
        reviser_2.set_decode_type("greedy")

    model.set_decode_type(
        "greedy" if opts.decode_strategy in ('bs', 'greedy') else "sampling",
        temp=softmax_temp)

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    results = []
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        batch = move_to(batch, device)

        start = time.time()
        with torch.no_grad():
            if opts.decode_strategy in ('sample', 'greedy'):
                if opts.decode_strategy == 'greedy':
                    assert width == 0, "Do not set width when using greedy"
                    assert opts.eval_batch_size <= opts.max_calc_batch_size, \
                        "eval_batch_size should be smaller than calc batch size"
                    batch_rep = 1
                    iter_rep = 1
                else:
                    batch_rep = width
                    iter_rep = 1
                assert batch_rep > 0
                # This returns (batch_size, iter_rep shape)


                tours, costs,costs_revised = model.sample_many(batch, opts,batch_rep=batch_rep, iter_rep=iter_rep,reviser=reviser,reviser_2=reviser_2)




        duration = time.time() - start

        if costs_revised is not None:
            results.append((costs.item(),costs_revised.item(), tours, duration))
        else:
            results.append((costs.item(),None, tours, duration))

    return results


if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="Filename of the dataset(s) to evaluate")

    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=1,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=2,
                        help="Softmax temperature (sampling or bs)")
    parser.add_argument('--seeder', type=str,default='pretrained_LCP/Seeder/seeder_tsp_50/epoch-99.pt')
    parser.add_argument('--reviser', type=str,default='pretrained_LCP/Reviser/reviser_20/epoch-99.pt')
    parser.add_argument('--reviser_2',default='pretrained_LCP/Reviser/reviser_10/epoch-99.pt',  type=str)
    parser.add_argument('--revision_len1', type=int, default=20, help='sub problem length for reviser 1')
    parser.add_argument('--revision_len2', type=int, default=10, help='sub problem length for reviser 2')
    parser.add_argument('--revision_iter1', type=int, default=20, help='number of iteration (I) for reviser 1')
    parser.add_argument('--revision_iter2', type=int, default=20, help='number of iteration (I) for reviser 1')
    parser.add_argument('--problem', default='tsp', type=str)
    parser.add_argument('--width', type=int, help='number of candidate solutions (M)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=10000, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Use multiprocessing to parallelize over multiple GPUs')

    opts = parser.parse_args()

    assert opts.o is None or (len(opts.datasets) == 1 and len(opts.width) <= 1), \
        "Cannot specify result filename with more than one dataset or more than one width"




    eval_dataset(opts.dataset_path, opts.width, opts.softmax_temperature, opts)
