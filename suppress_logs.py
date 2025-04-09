import os
import tensorflow as tf
import logging

# Set TensorFlow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR

# Disable other TensorFlow warnings and info messages
tf.get_logger().setLevel(logging.ERROR)

# Disable CUDA warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Suppress absl logging
logging.root.removeHandler(logging.root.handlers)
logging.basicConfig(level=logging.ERROR)

# Disable warning about computation placer
os.environ['TF_DISABLE_TENSOR_FLOAT_32_EXECUTION'] = '0'

# Specifically target XLA logging
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2'

# Disable algorithm picker logs
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# Direct TensorFlow to not use stderr for logging
os.environ['TF_SILENT_LOGGER'] = '1'