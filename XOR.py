from keras.models import Sequential
from keras.layers.core import Dense
from keras import backend as K

import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

import numpy as np


def export_model_for_mobile(model_name, input_node_name, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out', \
        model_name + '_graph.pbtxt')

    tf.train.Saver().save(K.get_session(), 'out/' + model_name + '.chkp')

    freeze_graph.freeze_graph('out/' + model_name + '_graph.pbtxt', None, \
        False, 'out/' + model_name + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out/frozen_' + model_name + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + model_name + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, [input_node_name], [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/tensorflow_lite_' + model_name + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())


xor = np.array([[0,0],[0,1],[1,0],[1,1],])
y_xor = np.array([[0],[1],[1],[0]])

model = Sequential(layers=[
    Dense(units=2, input_shape=(2,), activation='tanh'),
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')
model.optimizer.lr = 0.1
model.fit(xor, y_xor, epochs=100)

print("predicting [1, 0]: ")
print(model.predict_classes(np.asarray([[1, 0]])))
print("Predicting [0, 1]:")
print(model.predict_classes(np.asarray([[0, 1]])))
print("Predicting [0, 0]:")
print(model.predict_classes(np.asarray([[0, 0]])))
print("Predicting [1, 1]:")
print(model.predict_classes(np.asarray([[1, 1]])))

export_model_for_mobile('xor_nn', "dense_1_input", "dense_2/Sigmoid")
model.summary()
