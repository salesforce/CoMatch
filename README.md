## CoMatch: Semi-supervised Learning with Contrastive Graph Regularization (Salesforce Research)

This is a PyTorch implementation of the <a href="https://openreview.net/forum?id=KmykpuSrjcq">CoMatch paper</a>:
<pre>
@article{CoMatch,
	title={Semi-supervised Learning with Contrastive Graph Regularization},
	author={Junnan Li and Caiming Xiong and Steven C.H. Hoi},
	journal={},
	year={2021}
}</pre>

To perform semi-supervised learning on CIFAR-10 with 4 labels per class, run:
<pre>python Train_CoMatch.py --n-labeled 40 --seed 1 </pre> 

The results using different random seeds are:

seed| 1 | 2 | 3 | 4 | 5 | avg 
--- | --- | --- | --- | --- | --- | --- 
accuracy|93.71|94.10|92.93|90.73|93.97|93.09

For ImageNet experiments, see ./imagenet/

