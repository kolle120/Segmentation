import tensorflow as tf

def model_layers(image):
	with tf.name_scope('conv1') as scope:
		kernel1 = tf.Variable(tf.truncated_normal([3,3,32], dtype=tf.float32, stddev=1e-1), name='weights1')
		conv = tf.nn.conv2d(images, kernel1, [1,1,1,1], padding='SAME')
		conv1 = tf.nn.relu(conv, name=scope)
	with tf.name_scope('conv2') as scope:
		conv = tf.nn.conv2d(conv1, kernel1, [1,1,1,1], padding='SAME')
		conv2 = tf.nn.relu(conv, name=scope)
	pool1 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
	drop1 = tf.nn.dropout(pool1, 0.25, noise_shape=None, seed=None, name='dropout1')

	with tf.name_scope('conv3') as scope:
		kernel2 = tf.Variable(tf.truncated_normal([3,3,64], dtype=tf.float32, stddev=11e-1), name='weights2')
		conv = tf.nn.conv2d(drop1, kernel2, [1,1,1,1], padding='SAME')
		conv3 = tf.nn.relu(conv, name=scope)
	with tf.name_scope('conv4') as scope:
		conv = tf.nn.conv2d(conv3, kernel2, [1,1,1,1], padding='SAME')
		conv4 = tf.nn.relu(conv, name=scope)
	pool2 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
	drop2 = tf.nn.dropout(pool2, 0.25, noise_shape=None, seed=None, name='dropout2')

	with tf.name_scope('conv5') as scope:
		kernel3 = tf.Variable(tf.truncated_normal([1,1,512], dtype=tf.float32, stddev=1e-1), name='weights3')
		conv = tf.nn.conv2d(drop2, kernel3, [1,1,1,1], padding='SAME')
		conv5 = tf.nn.relu(conv, name=scope)

	with tf.name_scope('conv6') as scope:
		conv = tf.nn.conv2d(conv5, kernel3, [1,1,1,1], padding='SAME')
		conv6 = tf.nn.relu(conv, name=scope)

	with tf.name_scope('conv7') as scope:
		kernel4 = tf.Variable(tf.truncated_normal([1,1,1], dtype=tf.float32, stddev=1e-1), name='weights3')
		conv = tf.nn.conv2d(conv6, kernel4, [1,1,1,1], padding='SAME')
		conv7 = tf.nn.relu(conv, name=scope)

		final_image = tf.image.resize_images(conv7, 224, 224)