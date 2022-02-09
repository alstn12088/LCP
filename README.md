# Learning Collaborative Policies to Solve NP-hard Routing Problems (Code will be posted in 2022/02/16 with Pretrained Model)

Learhing collaborative policies (LCP) is new problem-solving strategy to tackle NP-hard routing problem such as travelling salesman problem. LCP uses existing competitive model such as Attention Model (AM). We have two main policies: seeder and reviser. Seeder searches full trajectories trained with proposed scaled entropy regularization. Reviser improves seeder's initial feasible candidate solutions in restricted solution space (i.e., partial solution). 



## Paper
This is official PyTorch code for our paper [Learning Collaborative Policies to Solve NP-hard Routing Problems](https://arxiv.org/abs/2110.13987) which has been accepted at [NeurIPS 2021](https://papers.nips.cc/paper/2021), cite our paper as follows:

```
@inproceedings{kim2021learning,
  title={Learning Collaborative Policies to Solve NP-hard Routing Problems},
  author={Kim, Minsu and Park, Jinkyoo and Kim, Joungho},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

## Thanks to

This code is originally implemented based on  [Attention Model](https://github.com/wouterkool/attention-learn-to-route) , which is source code of the paper   [Attention, Learn to Solve Routing Problems!](https://openreview.net/forum?id=ByxBFsRqYm) which has been accepted at [ICLR 2019](https://iclr.cc/Conferences/2019), cite as follows:

```
@inproceedings{
    kool2018attention,
    title={Attention, Learn to Solve Routing Problems!},
    author={Wouter Kool and Herke van Hoof and Max Welling},
    booktitle={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/forum?id=ByxBFsRqYm},
}
```

Our work designed collaborative polices (seeder and reviser), each policy is parameterized with Attention Model (AM), most of code configuration is same, except:

* New python file containing slightly modified neural architecture for reviser named "nets/attention_local.py".
* Modified "net/attention_model.py" to measure entropy of segment policy (see paper for detail).
* Modified "train.py" to add scaled entropy regularization term. 
* Modified 'sample_many' function in "functions.py" to modify solution design process with collaborative policies. 
* Modified 'eval.py' to modify solution design process.

## Important Remark

Our work is scheme using two collaborative polices to tackle problem complexity. Therefore, the AM is just example architecture to verify our idea. Please use our idea to state-of-the-art neural combinatorial optimization models to get higher performances.


## Dependencies

* Python>=3.7
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.1
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib 



