import numpy as np

# Communication to TensorFlow server via gRPC
import grpc
import tensorflow as tf

# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.contrib.util import make_tensor_proto

from os import listdir
from os.path import isfile, join

timeout = 60.0


channel = grpc.insecure_channel('localhost:8501')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

input_data = np.array([[2061, 318, 428, 30]], dtype='int32')
# Boiler-plate
request = predict_pb2.PredictRequest()

# Set request objects using the tf-serving `CopyFrom` setter method
request.model_spec.name = '0'
request.model_spec.signature_name = 'serving_default'
# This is correct (default constant).
request.inputs['input'].CopyFrom(make_tensor_proto(input_data,
                                                   shape=input_data.shape))

# Boiler-Plate
response = stub.Predict(request, timeout)

result = response.outputs['output']
print(tf.make_ndarray(result))
