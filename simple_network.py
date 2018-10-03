import numpy as np
import tensorflow as tf
import pandas as pd

INPUT_SHAPE = (28, 28)
NUM_NEURON_FC1 = 200
NUM_NEURON_FC2 = 512
NUM_NEURON_FC3 = 256
NUM_NEURON_FC4 = 128
NUM_NEURON_OUT = 10


def weight_var(shape, name=None):
	w = tf.truncated_normal(shape)
	return tf.Variable(w, trainable=True, name=name)


def bias_var(shape, name):
	b = tf.truncated_normal(shape=shape)
	return tf.Variable(b, trainable=True, name=name)


def fc_layer(inp, ch_in, ch_out, name='fc'):
	w = weight_var([ch_in, ch_out], name='W')
	b = bias_var([ch_out, ], name='b')
	act = tf.nn.relu(tf.matmul(inp, w) + b)
	# tf.summary.histogram("weights", w)
	# tf.summary.histogram("biases", b)
	# tf.summary.histogram("activations", act)
	return act


def fc_net(input_net):
	net = {}
	inshape = input_net.get_shape()
	flat = tf.reshape(input_net, shape=(-1, inshape[1] * inshape[2]))
	# flat = tf.layers.flatten(input_net)
	net['flat'] = flat
	fc1 = fc_layer(flat, INPUT_SHAPE[0] * INPUT_SHAPE[1], NUM_NEURON_FC1, name='fc1')
	net['fc1'] = fc1
	fc2 = fc_layer(fc1, NUM_NEURON_FC1, NUM_NEURON_FC2, name='fc2')
	net['fc2'] = fc2
	# fc3 = tf.nn.dropout(fc_layer(fc2, NUM_NEURON_FC2, NUM_NEURON_FC3, name='fc3'), keep_prob = 0.2)
	# net['fc3'] = fc3
	# fc4 = tf.nn.dropout(fc_layer(fc3, NUM_NEURON_FC3, NUM_NEURON_FC4, name='fc4'), keep_prob = 0.3)
	# net['fc4'] = fc4
	out = tf.nn.dropout(fc_layer(fc2, NUM_NEURON_FC2, NUM_NEURON_OUT, name='fc_out'), keep_prob = 0.5)
	net['out'] = out
	return net


def conv_labels(labels):
	n = labels.shape[0]
	train_labels = np.ndarray((n, NUM_NEURON_OUT))
	for i in range(n):
		for j in range(10):
			if labels[i] == j:
				train_labels[i, j] = 1
	return train_labels


if __name__ == "__main__":
	train_data_path = 'data/train.csv'
	test_data_path = 'data/test.csv'

	train = pd.read_csv(train_data_path)
	test = pd.read_csv(test_data_path)
	train_x = train.values[:, 1:]
	train_x = np.reshape(train_x, (42000, 28, 28))
	train_x = np.expand_dims(train_x, -1)
	train_y = train.values[:, 0]
	train_y = conv_labels(train_y)
	test = test.values

	print(train_x.shape, train_y.shape, test.shape)

	input_place = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_SHAPE[0], INPUT_SHAPE[1], 1))
	input_labels = tf.placeholder(dtype=tf.float32, shape=(None, NUM_NEURON_OUT))
	logits = fc_net(input_place)["out"]
	xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=logits))

	lr = 1e-3
	opt = tf.train.AdamOptimizer(learning_rate = lr)
	train_step = opt.minimize(xent)

	correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(input_labels, 1))
	acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	VALID_SIZE = 2000
	val_x = train_x[:VALID_SIZE]
	val_y = train_y[:VALID_SIZE]

	train_x_v = train_x[VALID_SIZE:]
	train_y_v = train_y[VALID_SIZE:]
	train_size = train_x_v.shape[0]
	batch_size = 5000
	batch_num = train_size // batch_size

	epochs = 10
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		accuracy = sess.run(acc, feed_dict={input_place: train_x, input_labels: train_y})
		print('Ini1 acc %f' % accuracy)
		for i in range(epochs):
			batch_start = 0
			for batch in range(batch_num):
				sess.run(train_step, 
					feed_dict={input_place : train_x_v[batch_start: batch_start+batch_size], 
					input_labels: train_y_v[batch_start: batch_start+batch_size]})
				val_acc = sess.run(acc, feed_dict={input_place: val_x, input_labels: val_y})
				# print("Epoch %d, (%d, %d), val_acc=%f" % (i, batch_start, batch_start + batch_size - 1, val_acc))
				batch_start = (batch_start + batch_size) % train_size

			# sess.run(train_step, feed_dict={input_place: train_x, input_labels: train_y})
			accuracy = sess.run(acc, feed_dict={input_place: val_x, input_labels: val_y})
			print('Epoch %d > val_acc = %f' % (i, accuracy))
