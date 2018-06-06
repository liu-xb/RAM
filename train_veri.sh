caffe/build/tools/caffe train -solver prototxt/solver.prototxt -weights ~/caffemodel/vgg-cnn-m/VGG_CNN_M.caffemodel -gpu 0

caffe/build/tools/caffe train -solver prototxt/solver_finetune.prototxt -weights snapshot/veri-attribute_iter_80000.caffemodel -gpu 0

