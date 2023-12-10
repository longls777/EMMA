# EMMA
Source code of 《Fusion Makes Perfection: An Efficient Multi-Grained Matching Approach for Zero-Shot Relation Extraction》

## Environments
> python: 3.7.16<br>
> torch: 1.10.1+cu111<br>
> torchvision: 0.11.2+cu111<br>
> numpy: 1.21.6<br>


## Datasets
You can download the dataset used in this work via the following google drive link,and then store them in the data/fewrel and data/wikizsl paths respectively.

[FewRel (Xu et al., 2018)](https://drive.google.com/file/d/1PgSTaEEUxsE-9lhQan3Yj91pzLhxv7cT/view?usp=sharing)

[WikiZSL (Daniil Sorokin and Iryna Gurevych, 2017)](https://drive.google.com/file/d/1kGmhlpTTq8UmIUPZ2CSIruWWsi_l_ERH/view?usp=share_link)

## Train&Inference
You can easily run training as well as inference with the following scripts

> bash run.sh<br>

## Acknowledgement
Our implementation drew inspiration from parts of the codes from [RE-Matching](https://github.com/zweny/RE-Matching), and we appreciate their provision of open-source code.
