caffe/build/tools/caffe train -solver prototxt/solver_CN.prototxt -weights ~/caffemodel/vgg-cnn-m/VGG_CNN_M.caffemodel -gpu 0

caffe/build/tools/caffe train -solver prototxt/solver_CN_finetune.prototxt -weights snapshot/veri-CN_iter_60000.caffemodel -gpu 0

caffe/build/tools/caffe train -solver prototxt/solver_BN.prototxt -weights snapshot/veri-CN-finetune_iter_60000.caffemodel -gpu 0

caffe/build/tools/caffe train -solver prototxt/solver_BN_finetune.prototxt -weights snapshot/veri-BN_iter_60000.caffemodel -gpu 0

caffe/build/tools/caffe train -solver prototxt/solver_BN+R.prototxt -weights snapshot/veri-BN-finetune_iter_60000.caffemodel -gpu 0

caffe/build/tools/caffe train -solver prototxt/solver_BN+R_finetune.prototxt -weights snapshot/veri-BN+R_iter_60000.caffemodel -gpu 0

caffe/build/tools/caffe train -solver prototxt/solver_RAM.prototxt -weights snapshot/veri-BN+R-finetune_iter_60000.caffemodel -gpu 0

caffe/build/tools/caffe train -solver prototxt/solver_RAM_finetune.prototxt -weights snapshot/veri-RAM_iter_60000.caffemodel -gpu 0

echo "The final model is \"veri-RAM-finetune_iter_60000.caffemodel\". "