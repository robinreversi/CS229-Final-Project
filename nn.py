import pandas as pd
import numpy as np
import tensorflow as tf

train_data = pd.read_csv('../train_10-1000_binary.csv')
dev_data = pd.read_csv('../dev_10-1000_binary.csv')
test_data = pd.read_csv('../test_10-1000_binary.csv')

learning_rate = 0.01
num_steps = 100
display_step = 10
# Network Parameters
n_hidden_1 = 4685 # 1st layer number of neurons
n_hidden_2 = 4685
num_input = 4685 # MNIST data input (img shape: 28*28)
num_classes = 12 # MNIST total classes (0-9 digits)


def onehot(y):
    ret = np.zeros((len(y),num_classes))
    for i in range(len(y)):
        num = y[i]
        ret[i,int(num)] = 1.0
    return ret

train_x = np.array(train_data.iloc[:, 1:])
train_y = onehot(np.array(train_data['0'].values))
dev_x = np.array(dev_data.iloc[:, 1:])
dev_y = onehot(np.array(dev_data['0'].values))
test_x = np.array(test_data.iloc[:, 1:])
test_y = onehot(np.array(test_data['0'].values))

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = (tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer_2 = (tf.add(tf.matmul(layer_1,weights['h2']), biases['b2']))
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y) + 0.001 * tf.nn.l2_loss(weights['h1'])+ 0.001 * tf.nn.l2_loss(weights['h2'])+ 0.001 * tf.nn.l2_loss(weights['out'])
                         +0.001 * tf.nn.l2_loss(biases['b1'])+ 0.001 * tf.nn.l2_loss(biases['b2'])+ 0.001 * tf.nn.l2_loss(biases['out']))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        inds = np.random.permutation(train_x.shape[0])[:200]
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: train_x[inds,:], Y: train_y[inds,:]})
        if step % display_step == 0:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: train_x,
                                                                    Y: train_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                    "{:.4f}".format(loss) + ", Training Accuracy= " + \
                    "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: dev_x, Y: dev_y}))
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_x, Y: test_y}))