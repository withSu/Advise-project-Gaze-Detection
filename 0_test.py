import tensorflow as tf

print("TensorFlow에서 GPU 사용 가능 여부:", tf.config.list_physical_devices('GPU'))
print("cuDNN 사용 가능 여부:", tf.test.is_built_with_cuda())
print("cuDNN 버전:", tf.sysconfig.get_build_info()["cudnn_version"])
