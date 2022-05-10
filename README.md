# [RAM: A Region-Aware Deep Model for Vehicle Re-Identification (ICME'18)](https://ieeexplore.ieee.org/document/8486589)

![](icme2018.jpg)

Here are codes of my ICME 2018 paper, "RAM: A Region-Aware Deep Model for Vehicle Re-Identification".<br>
If you find this help, please kindly cite our paper:<br>
```
@inproceedings{icme-ram-liu,
  Author = {Liu, Xiaobin and Zhang, Shiliang and Huang, Qingming and Gao, Wen},<br>
  Booktitle = {ICME},
  Title = {RAM: A Region-Aware Deep Model for Vehicle Re-Identification},
  Year = {2018}
}
```

## Train the model

You can simply train a RAM on VeRi by running:<br>
```
sh train_veri.sh
```
The final model is saved as "snapshot/veri-RAM-finetune_iter_60000.caffemodel".
Our models and extracted features on VeRi can be downloaded from [Baidu Disk](https://pan.baidu.com/s/1TMF0crG6SupWsfzEOFSr6A?pwd=87dn 
) with pass word: 87dn , or [Google Drive](https://drive.google.com/drive/folders/15tA1biAG2-TAoar2eI9iVNragJylzAOl?usp=sharing).

We provide a new caffe layer to sample mini-batch. Please refer to "prototxt/train_RAM.prototxt" for an example of usage.

## Evaluate the performance

We provide an evaluate script, "evaluate.m", on VeRi following https://github.com/VehicleReId/VeRidataset <br>

We provide a tools in caffe to extract features and write features to binary files. We also provide a tools to read features from binary file, "read_code.m", and a tools to normalize features, "norm_code.m". An example of the usage of these tools can be found in "evaluate.m".

## Contact me
Email: xbliu DOT vmc AT pku.edu.cn <br>
Homepage: https://liu-xb.github.io <br>
Please feel free to contact me if you have any question.
