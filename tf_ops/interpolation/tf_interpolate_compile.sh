# TF1.2
# g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

TF_BASE='/home/subbotnik/pointnet2/env/local/lib/python2.7/site-packages/tensorflow/include/tensorflow'
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

# TF1.4
g++ -v -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I $TF_INC/external/nsync/public -I $TF_INC -L $TF_LIB -ltensorflow_framework -I /usr/local/cuda-9.0/include -lcudart -L /usr/local/cuda-9.0/lib64/ -L $TF_BASE -O2 -D_GLIBCXX_USE_CXX11_ABI=0
