# S3AI
The official code repository of ["Interpretable antibody-antigen interaction prediction by introducing route and priors guidance"](https://www.biorxiv.org/content/10.1101/2024.03.09.584264v1).
![Our pipeline](./figs/fig1-v2.png)
## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Data](#data)
- [Model inference](#model-inference)
- [Model training](#model-training)
- [Citation](#citation)
- [License](#license)

## Overview
With the application of personalized and precision medicine, more precise and efficient antibody drug development technology is urgently needed. Identification of antibody-antigen interactions is key to antibody engineering. The time-consuming and expensive nature of wet-lab experiments calls for efficient computational methods. Previous deep-learning-based computing methods for antibody-antigen interaction prediction are distinctly divided into two categories: structure-based and sequence-based. Taking into account the non-overlapping advantage of these two major categories, we propose an interpretable antibody-antigen interaction prediction method, S3AI, that bridges structures to sequences through structural information distillation. Furthermore, non-covalent interactions are modeled explicitly to guide neural networks in understanding the underlying patterns in antigen-antibody docking. Supported by the two innovative designs mentioned above, S3AI significantly and comprehensively surpasses the state-of-the-art models. S3AI maintains excellent robustness when predicting unknown antibody-antigen pairs, surpassing specialized prediction methods designed for out-of-distribution generalization in fair comparisons. More importantly, S3AI captures the universal pattern of antibody-antigen interactions, which not only identifies the CDRs responsible for specific binding to the antigen but also unearthed the importance of CDR-H3 for the interaction. The implicit introduction of knowledge of structure modality and the explicit modeling of chemical constraints build a 'sequence-to-function' route, thereby facilitating S3AI's understanding of complex molecular interactions through providing route and priors guidance. S3AI, which does not require structure input, is suitable for large-scale, parallelized antibody optimization and screening while outperforming state-of-the-art prediction methods. It helps to quickly and accurately identify potential candidates in the vast antibody space, thereby accelerating the development process of antibody drugs.
## Hardware requirements

The experiments are tested and conducted on one Tesla V100 (32GB).

## Installation

We highly recommand that you use Anaconda for Installation
```
conda create -n S3AI
conda activate S3AI
pip install -r requirements.txt
pip install fair-esm pyaml==21.10.1
```

## Data
The SARS-CoV-2 IC50 data is in the `data` folder.
* `data/updated_processed_data.csv` is the paired Ab-Ag data.
* `data/Ag_sequence.csv` is the Ag sequence data.

## Model inference 
Download the checkpoint of S3AI and modify the paths in the code.
| Content  | Link   |
| ----- | ----- |
| Checkpoint on SARS-CoV-2 | [link](https://figshare.com/ndownloader/files/44970310) |
| Checkpoint on HIV cls | [link](https://figshare.com/ndownloader/files/45053224) |
| Checkpoint on HIV reg | [link](https://figshare.com/ndownloader/files/45104590) |

To test S3AI on SARS-CoV-2 IC50 test data, please run
```
python main.py --config=configs/test_on_sarscov2.yml
```

To test S3AI on HIV test data for classification, please run
```
python main.py --config=configs/test_on_HIV_cls.yml
```

To test S3AI on HIV test data for regression, please run
```
python main.py --config=configs/test_on_HIV_reg.yml
```

## Model training
To train S3AI on downstream tasks from scratch, please run
```
python main.py --config=configs/train_on_sarscov2.yml
python main.py --config=configs/train_on_HIV_cls.yml
python main.py --config=configs/train_on_HIV_reg.yml
```

## Attribution evaluation

### Attribution of entire CDRs and FRs

To evaluate the attribution of the entire CDRs and FRs to the prediction output on test data, please run:

```python
python attribution_entire.py --config=configs/test_attribution.yml
```

#### File Description

You will save a file in the format `shapley_phi_st{args.sample_st}_samplenum{args.sample_num}.npz`, where **sample_st** indicates the starting index of the stored samples in the training set, and **sample_num** indicates the number of samples for which attributions are computed. 

This file stores the following elements:

- **v_N:** Predicted output of the network on each sample
- **phi:** Attribution value of each FR (Framework Region) and the entire CDRs (Complementarity-determining regions)
- **player:** Partition of the variables. For each variable, it saves a tuple $(l, r)$, where the tuple represents the position of the corresponding FR on the antibody sequence. Specifically, $(-1, -1)$ indicates the entire CDRs.
- **CDR_H:** Saves three tuples $(l, r)$ representing the positions of CDR-H1, CDR-H2, and CDR-H3 on the antibody sequence.
- **CDR_L:** Saves three tuples $(l, r)$ representing the positions of CDR-L1, CDR-L2, and CDR-L3 on the antibody sequence.
- **ic50:** True IC50 values of each sample.
- **cls_label:** True classification results of each sample.

### Attribution of each CDR

To evaluate the attribution of each CDR  on test data, please run:

```python
python attribution_eachCDR.py --config=configs/test_attribution.yml
```

#### File Description

You will save a file in the format `inner_phi.npz`. This file stores the following elements:

- **v_N:** attribution value of the entire CDRs under different masks
- **phi:** Attribution value of each CDR (Complementarity-determining region)
- **player:** Partition of the variables. For each variable, it saves a tuple $(l, r)$, where the tuple represents the position of the corresponding CDR on the antibody sequence. 
- **CDR_H:** Saves three tuples $(l, r)$ representing the positions of CDR-H1, CDR-H2, and CDR-H3 on the antibody sequence.
- **CDR_L:** Saves three tuples $(l, r)$ representing the positions of CDR-L1, CDR-L2, and CDR-L3 on the antibody sequence.

## Citation

If you find this code or the models useful in your research, please cite:
```
@article{liu2024interpretable,
  title={Interpretable antibody-antigen interaction prediction by bridging structure to sequence},
  author={Liu, Yutian and Nie, Zhiwei and Chen, Jie and Zheng, Xinhao and Fu, Jie and Liu, Zhihong and Liu, Xudong and Xu, Fan and Huang, Xiansong and Zhang, Wen-Bin and others},
  journal={bioRxiv},
  pages={2024--03},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
## License

This project is licensed under the [MIT License](LICENSE).

