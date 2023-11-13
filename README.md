# Whale whistle contour extactor trained on pseudo-label
This is the official repository of our paper 
"[Using deep learning to track time× frequency whistle contours of toothed whales without human-annotated training data](https://pubs.aip.org/asa/jasa/article/154/1/502/2904481)" 
published on The Journal of the Acoustical Society of America in 2023. 

### Updates
[2023-11-12] Model Training code uploaded. 
### To Dos
- Pseudo-label generation code
- Model evaluation code

### <a name="dependency"></a> Dependency
* Ubuntu ≥ 14.04
* Python ≥ 3.6.8
* CUDA ≥ 10.0
* Pytorch ≥ 1.4.0

Other python libraries:
> ```bash
> pip install -r requirements.txt
> ```

### <a name="Raw data"></a> Raw data
we retrieved the raw data from [mobysound.org](mobysound.org). 
> **_NOTE:_** It seems that the web server of mobysound are experincing problems. 
We will update this section when the raw data are available. 

Specifically, we used the following files:
- 5th_DCL_data_bottlenose.zip
- 5th_DCL_data_common.zip
- 5th_DCL_data_melon-headed.zip
- 5th_DCL_data_spinner.zip


You may extract the raw files using:
> ```bash
> unzip [filename].zip
> ```

### <a name="Data split"></a> Data split
We split the raw data into three parts:
- annotated training data
- unannotated training data
- testing data

### <a name="Data processing"></a> Data processing

### <a name="Pseudo label generation"></a> Pseudo label generation
1. Silbido
2. SMC-PHD

### <a name="Model training"></a> Model training
After you have the training data and testing (h5 files), you may write the path to these
data in the following files (each h5 file covers one line in the txt file):
- plwe_pytorch/train_pl_pos.txt: path to the data of positive samples, which contains 
whistles indicated by pseudo-label
- plwe_pytorch/train_pl_neg.txt: path to the data of negative samples, which contains 
no whistles according to pseudo-label
- plwe_pytorch/test.txt: path to testing data

You may configure the training parameters in "plwe_pytorch/run.sh" whose contents are:
> ```bash
> CUDA_VISIBLE_DEVICES=0 python train.py --lamda 1.000 --lamda_pos 4.000 --lamda_neg 0.000   --gamma_pos 2 --gamma_neg 0
> ```
where 
- lamda is the weight for L_base in Eq.3 or Eq.5 in our paper
- lamda_recall and gamma_recall is the lamda and gamma in Eq.3, respectively
- lamda_prec and gamma_prec is the lamda and gamma in Eq.5, respectively

You may train the model with following code:
> ```bash
> cd plwe_pytorch
> bash run.sh
> ```


### <a name="Model Evaluation"></a> Model Evaluation


