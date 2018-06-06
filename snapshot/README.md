# RAM (ICME2018)

This is the code of our ICME 2018 paper, "RAM: A Region-Aware Deep Model for Vehicle Re-Identification".<br>
If you find this help, please kindly cite our paper:<br>
```
@inproceedings{cime-ram-liu,<br>
  Author = {Liu, Xiaobin and Zhang, Shiliang and Huang, Qingming and Gao, Wen},<br>
  Booktitle = {ICME},<br>
  Title = {RAM: A Region-Aware Deep Model for Vehicle Re-Identification},<br>
  Year = {2018}<br>
}
```

## Train the model

You can simply train a RAM on VeRi by running:<br>
```
./train_veri.sh
```
Note that, we actually train the RAM in a step-by-step manner as we present in our paper. So that, directly train the whole RAM may result in performance a little lower than we report in our paper. Our model on VeRi can be downloaded from [here](https://pan.baidu.com/s/17fnjp1fAvWNmIrF1wzkQqg).

We provide a new caffe layer to sample mini-batch. Please refer to "prototxt/train_attribute.prototxt" for an example of usage.

## Evaluate the performance

We provide an evaluate script, "evaluate.m", on VeRi following https://github.com/VehicleReId/VeRidataset <br>

We provide a tools in caffe to extract features and write features to binary files. We also provide a tools to read features from binary file, "read_code.m", and a tools to normalize features, "norm_code.m". An example of the usage of these tools can be found in "evaluate.m".

## Contact me
Email: xbliu.vmc@pku.edu.cn <br>
Homepage: https://liu-xb.github.io <br>
Please feel free to contact me if you have any question.