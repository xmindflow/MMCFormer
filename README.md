# [MMCFormer: Missing Modality Compensation Transformer for Brain Tumor Segmentation](https://openreview.net/forum?id=PD0ASSmvlE) (Accepted for oral presentation at MIDL 2023 conference)

![Diagram of the proposed method](https://github.com/mindflow-institue/MMCFormer/blob/main/images/method.png)
In this research work, to address missing modality issue in the MRI-based semantic segmentation tasks, we propose MMCFormer, a novel missing modality compensation network. Our strategy builds upon 3D efficient transformer blocks and uses a co-training strategy to effectively train a missing modality network. To ensure feature consistency in a multi-scale fashion, MMCFormer utilizes global contextual agreement modules in each scale of the encoders. Furthermore, to transfer modality-specific representations, we propose to incorporate auxiliary tokens in the bottleneck stage to model interaction between full and missing-modality paths. On top of that, we include feature consistency losses to reduce the domain gap in network prediction and increase the prediction reliability for the missing modality path.

## Updates
- Paper accepted in MIDL2023

This code has been implemented in Python language using Pytorch library and tested in Ubuntu OS, though should be compatible with related environment. following environment and Library needed to run the code:

- Python 3
- Pytorch


## Run Demo
For training deep model and evaluating the BraTA 2018 dataset set, follow the bellow steps:</br>
1- Download the BraTS 2018 train dataset from [this](https://www.kaggle.com/sanglequang/brats2018) link and extract it inside the `dataset_BraTS2018` folder. </br>
2- Run `train.py` for training the model. </br>
3- For performance calculation and producing segmentation results, run `evaluation.ipynb`.</br>

Notice: our implementation uses the SMU codes: https://github.com/rezazad68/smunet

## Citation
If this code helps with your research please consider citing the following paper:

```python
@inproceedings{
  karimijafarbigloo2023mmcformer,
  title={{MMCF}ormer: Missing Modality Compensation Transformer for Brain Tumor Segmentation},
  author={Sanaz Karimijafarbigloo and Reza Azad and Amirhossein Kazerouni and Saeed Ebadollahi and Dorit Merhof},
  booktitle={Medical Imaging with Deep Learning},
  year={2023},
  url={https://openreview.net/forum?id=PD0ASSmvlE}
}
```
