# DU-VAE
This is the pytorch implementation of the paper "Regularizing Variational Autoencoder with Diversity and Uncertainty Awareness"

### Acknowledgements
Our code is mainly based on this public [code](https://github.com/jxhe/vae-lagging-encoder). 
Very thanks for its authors.

### Requirements
* Python >= 3.6
* Pytorch >= 1.5.0


### Data
Datastes used in this paper can be downloaded in this [link](https://drive.google.com/drive/folders/13sMpOJLFkROPxaIBKl8NUSOEvQjUQyq_?usp=sharing), with the specific license if that is not based on MIT License. 
### Usage
Example script to train DU-VAE on text data:
```angular2
python text.py --dataset yelp \
 --device cuda:0  \
--gamma 0.5 \
--p_drop 0.2 \
--delta_rate 1 \
--kl_start 0 \
--warm_up 10
```
Example script to train DU-VAE on image data:
```angular2
python3.6 image.py --dataset omniglot \
 --device cuda:3 \
--kl_start 0 \
--warm_up 10 \
--gamma 0.5  \
--p_drop 0.1 \
--delta_rate 1 \
--dataset omniglot
```
Example script to train DU-IAF, a variant of DU-VAE,  on text data:
```angular2
python3.6 text_IAF.py --device cuda:2 \
--dataset yelp \
--gamma 0.6 \
--p_drop 0.3 \
--delta_rate 1 \
--kl_start 0 \
--warm_up 10 \
--flow_depth 2 \
--flow_width 60
```
Example script to train DU-IAF on image data:
```angular2
python3.6 image_IAF.py --dataset omniglot\
  --device cuda:3 \
--kl_start 0 \
--warm_up 10 \
--gamma 0.5 \
 --p_drop 0.15\
 --delta_rate 1 \
--flow_depth 2\
--flow_width 60 
```
Here,
* `--dataset` specifies the dataset name, currently it supports `synthetic`, `yahoo`, `yelp` for `text.py` and `omniglot` for `image.py`.
* `--kl_start` represents starting KL weight (set to 1.0 to disable KL annealing)
* `--warm_up` represents number of annealing epochs (KL weight increases from `kl_start` to 1.0 linearly in the first `warm_up` epochs)
* `--gamma` represents the parameter $\gamma$ in our Batch-Normalization approach, which should be more than 0 to use our model.
* `--p_drop` represents the parameter $1-p$ in our Dropout approach, which denotes the percent of data to be ignored and should be ranged in (0,1).
* `--delta_rate` represents the hyper-parameter $\alpha$ to controls the min value of the variance $\delta^2$
* `--flow_depth` represents number of MADE layers used to implement DU-IAF.
* `--flow_wdith` controls the hideen size in each IAF block, where we set the product between the value and the dimension of $z$ as the hidden size. 
For example, when we set `--flow width 60` with the dimension of $z$ as 32, the hidden size of each IAF block is 1920. 

### Reference
If you find our methods or code helpful, please kindly cite the paper: 
```angular2
@inproceedings{shen2021regularizing,
  title={Regularizing Variational Autoencoder with Diversity and Uncertainty Awareness},
  author={Shen, Dazhong  and Qin, Chuan and Wang, Chao and Zhu, Hengshu and Chen, Enhong and Xiong, Hui},
  booktitle={Proceedings of the 30th International Joint Conference on Artificial Intelligence (IJCAI-21)},
  year={2021}
}
``` 