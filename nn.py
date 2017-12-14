import pandas as pd
import numpy as np
import tensorflow as tf

train_data = pd.read_csv('../train_10-1000_norm.csv')
dev_data = pd.read_csv('../dev_10-1000_norm.csv')
test_data = pd.read_csv('../test_10-1000_norm.csv')

lst = range(0, 1000, 50)

learning_rate = 0.1
num_steps = 1000
display_step = 10



def onehot(y):
    ret = np.zeros((len(y),12))
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

num_input = len(train_x[0,:])
n_hidden_1 = 360
#n_hidden_2 = 120
num_classes = 12

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    #'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    #'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    #layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1,weights['h2']), biases['b2']))
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y) + 0.05 * tf.nn.l2_loss(weights['out']) + 0.05 * tf.nn.l2_loss(weights['h1']) + 0.05 * tf.nn.l2_loss(biases['b1'])# + 0.01 * tf.nn.l2_loss(weights['h2']) + 0.01 * tf.nn.l2_loss(biases['b2'])
                         + 0.05 * tf.nn.l2_loss(biases['out']))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    k = 200
    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        inds = np.random.permutation(train_x.shape[0])[:k]
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: train_x[inds,:], Y: train_y[inds,:]})
        if step % display_step == 0:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: train_x,
                                                                    Y: train_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                    "{:.4f}".format(loss) + ", Training Accuracy= " + \
                    "{:.3f}".format(acc))
            if loss < 1:
                learning_rate = 0.01
            elif loss < 5:
                learning_rate = 0.05
        if step in lst:
            pred,dev_acc = sess.run([prediction,accuracy], feed_dict={X: dev_x, Y: dev_y})
            top2 = np.argsort(pred)[:,[-1,-2]]
            count = 0.0
            for j in range(dev_y.shape[0]):
                if np.argmax(dev_y[j,:]) in top2[j,:]:
                    count+=1
            top2_acc = count / dev_y.shape[0]
            print("Dev Accuracy:", dev_acc)
            print("Dev Top 2 Accuracy:", top2_acc)
        if step > 350:
            k = train_x.shape[0]

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    pred, dev_acc = sess.run([prediction, accuracy], feed_dict={X: dev_x, Y: dev_y})
    top2 = np.argsort(pred)[:, [-1, -2]]
    count = 0.0
    for j in range(dev_y.shape[0]):
        if np.argmax(dev_y[j, :]) in top2[j, :]:
            count += 1
    top2_acc = count / dev_y.shape[0]
    print("Dev Accuracy:", dev_acc)
    print("Dev Top 2 Accuracy:", top2_acc)