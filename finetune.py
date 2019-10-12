from datetime import datetime
import sys
import tensorflow as tf
import numpy as np

from model import Model
from dataset import Dataset
from network import *
import tensorflow.contrib.slim as slim


def main():
    if len(sys.argv) != 4:
        print('Usage: python3 finetune.py train_file test_file weight_file')
        return

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    weight_file = sys.argv[3]

    # checkpoint save path   
    checkpoint_path = '/path/to/saved/model/vggface16_finetuned.ckpt'
    # writer path        
    train_writer_path = '/path/to/logs/train'
    test_writer_path = '/path/to/logs//test'

    # Learning params    
    learning_rate_init = 0.001   # 0.001
    decay_steps = 10000
    decay_rate = 0.5

    # Train and dispaly params
    training_iters = 60000
    batch_size = 50         #36
    display_step = 20
    test_step = 1000
    save_step = 1000

    # Network params
    n_classes = 10575
    keep_rate = 0.5  # 1.0

    # Graph input
    x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])    #input of vgg is 224*224, input of alexnet is 227*227
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_var = tf.placeholder(tf.float32)

    # Model
    # tf.reset_default_graph()
    pred = Model.vgg16(x, keep_var, n_classes)

    # Loss
    with tf.name_scope('loss'):   
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    # Optimizer           
    global_step = tf.Variable(0, trainable=False, name='global_step')
    with tf.name_scope('learning_rate'):
        learning_rate = tf.train.exponential_decay(learning_rate_init, global_step, decay_steps, decay_rate, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    # Evaluation
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Init
    init = tf.global_variables_initializer()

    # Create a summary to monitor learning_rate, loss, accuracy  
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)

    merged_summary_op = tf.summary.merge_all()

    # Load dataset
    dataset = Dataset(train_file, test_file)

    # Create a saver
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)  # ADD
	
    # Launch the graph
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        print('Init variable')
        sess.run(init)

        print("Model restored from file: %s" % weight_file)
        load_with_skip(weight_file, sess, ['fc8'])  # Skip weights from fc8
        print('Start training')

        # write logs to Tensorboard  
        train_writer = tf.summary.FileWriter(train_writer_path, sess.graph)
        test_writer = tf.summary.FileWriter(test_writer_path, sess.graph)

        step = 1
        while step < training_iters:
            batch_xs, batch_ys = dataset.next_batch(batch_size, 'train')
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_var: keep_rate})

            summary = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys, keep_var: keep_rate})   
            # Write logs at every iteration
            train_writer.add_summary(summary, step)                                                    
           
            # Display testing status
            if step % test_step == 0:
                test_acc = 0.
                test_count = 0
                t_loss = 0
                for _ in range(dataset.test_size // batch_size):
                    batch_tx, batch_ty = dataset.next_batch(batch_size, 'test')
                    acc = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ty, keep_var: 1.})
                    b_loss = sess.run(loss, feed_dict={x: batch_tx, y: batch_ty, keep_var: 1.})
                    summary = sess.run(merged_summary_op, feed_dict={x: batch_tx, y: batch_ty, keep_var: 1.})    # ADD
                    test_writer.add_summary(summary, step)                                              # ADD
                    test_acc += acc
                    t_loss += b_loss
                    test_count += 1
                test_acc /= test_count
                t_loss/= test_count
                print('{} Iter {}: Testing Accuracy = {:.4f}, Test loss = {:.4f}'.format(datetime.now(), step, test_acc,t_loss), file=sys.stderr)

            # Display training status
            if step % display_step == 0:
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.})
                batch_loss = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.})
                lr = sess.run(learning_rate)
                print('{} Iter {}: Training Loss = {:.4f}, Accuracy = {:.4f}, lr = {:.4f}'.format(datetime.now(), step, batch_loss, acc, lr), file=sys.stderr)

            # Save model 	  
            if step % save_step == 0:
                saver.save(sess, checkpoint_path, global_step=step)
     
            step += 1
        print('Finish!')


if __name__ == '__main__':
    main()
