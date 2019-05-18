# merge_bn-caffe
In caffe framwork，this .py can merge bn and scale layer into conv layer，which can speed up test。

You need prepare deploy.prototxt(it must includes batchnorm and scale layer),.caffemodel。

So in the terminal,you should put it:$pyhon merge_bn.py deploy.prototxt XXX.caffemodel
