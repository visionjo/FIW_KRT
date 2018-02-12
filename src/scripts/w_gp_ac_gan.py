from six.moves import xrange  # pylint: disable=redefined-builtin
import src.data.preprocessing as dp
import tensorflow as tf
import argparse
import numpy as np

"""updated @ 2018-02-09"""


# Define Random Initilization
def xavier_init(size, std=1):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev * std)


def generator(z, c, way_Gw, opt_nml, Wz, Wc, alpha=65):
    # inputs = tf.concat(axis=1, values=[z, c])
    if way_Gw == 1:
        G_prob = tf.matmul(z, tf.diag(Wz)) + tf.matmul(c, Wc)
    elif way_Gw == 0:
        G_prob = tf.matmul(z, Wz) + tf.matmul(c, Wc)
    if opt_nml == 1:
        G_prob = G_prob / tf.norm(G_prob) * alpha
    G_prob = tf.nn.relu(G_prob)
    return G_prob


def discriminator(X, Wg, Ws_bs, Ws_nl):
    Og = tf.nn.sigmoid(tf.matmul(X, Wg))
    Os = tf.matmul(X, tf.concat([Ws_bs, Ws_nl], 1))
    return Og, Os


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def cross_entropy(logit, y):
    return -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))


def w_gp_ac_gan(cmd):
    lambda2 = cmd.w  # weight between Wc*c and Wz*z

    """Different Ways to Initialize the weights for Generator"""
    G_w = ["random+" + str(lambda2) + "weight", "random vector" + str(lambda2) + "weight"]
    Opt = ["Momentum", "Adam", "RMS", "SGD"]

    way_Gw = cmd.i  # 0: Wz is matrix; 1: Wz is vector
    way_opt = 0  # choose optimizor
    wgan = cmd.g  # 0: without WGAN; 1: with WGAN
    wz_s = cmd.s  # scale of Wz
    opt_nml = cmd.n  # 0: without Normalization; 1: with normalization

    # Parameter
    z_dim = 512
    X_dim = 512
    y_dim = 21000

    SEED = 66478
    BATCH_SIZE = 512
    num_epochs = 20
    LAMBDA = 1e1
    num_classes = 21000

    # prepare index and shuffle file

    trSf = dp.readfile("/data/21K/TrainValData_300Max-ps.train.r100.shuffle")
    vlSf = dp.readfile("/data/21K/TrainValData_300Max-ps.val.base.shuffle")
    tsSf = dp.readfile("/data/21K/TrainValData_300Max-ps.val.lowshot.shuffle")
    feaIdx = dp.readfile("/data/21K/TrainValData_300Max-ps.LineNumber.resnet34-b.0.pool5.lineidx")
    lblIdx = dp.readfile("/data/21K/TrainValData_300Max-ps.LineNumber.resnet34-b.0.pool5.label.lineidx")
    tsvFea = "/data/21K/TrainValData_300Max-ps.LineNumber.resnet34-b.0.pool5.tsv"
    tsvLbl = "/data/21K/TrainValData_300Max-ps.LineNumber.resnet34-b.0.pool5.label.tsv"

    train_size = len(trSf)

    # Define Input Variables
    X = tf.placeholder(tf.float32, shape=[None, X_dim])
    y = tf.placeholder(tf.float32, shape=[None, y_dim])
    z = tf.placeholder(tf.float32, shape=[None, z_dim])

    # Initialize Weights of Class center
    weights = np.array(np.load('/data/weights/Wg_class_center.npy'), dtype=np.float32)
    Wc = tf.Variable(lambda2 * weights, name="G2")

    if way_Gw == 0:
        Wz = tf.Variable(xavier_init([z_dim, X_dim], wz_s), name="G1")
    elif way_Gw == 1:
        Wz = tf.Variable(wz_s * xavier_init([X_dim], 1), name="G1")

    """Initialization for Discriminator: Binary Discriminatot & Softmax Classifier"""
    Wd = tf.Variable(xavier_init([X_dim, 1], 1), name="Dg")

    W_sf = np.load('/data/weights/Wc_iter100000.npy').transpose()

    Ws_nl = tf.Variable(W_sf[:, 20000:21000], name="Dc1")
    Ws_bs = tf.Variable(W_sf[:, 0:20000], name="Dc2")

    theta_G = [Wz, Wc]
    theta_D = [Wd, Ws_nl]

    G_sample = generator(z, y, way_Gw, opt_nml, Wz, Wc)

    D_real, C_real = discriminator(X, Wd, Ws_bs, Ws_nl)
    D_fake, C_fake = discriminator(G_sample, Wd, Ws_bs, Ws_nl)

    # Cross entropy aux loss
    Cr_loss = cross_entropy(C_real, y)
    Cf_loss = cross_entropy(C_fake, y)

    C_loss = Cr_loss + Cf_loss

    # GAN D loss
    D_loss = tf.reduce_mean(tf.log(D_real)) + tf.reduce_mean(tf.log(1. - D_fake))

    # See if use W-GAN
    if wgan == 1:
        # Gradient Panelty
        alpha = tf.random_uniform(shape=[BATCH_SIZE, 1], minval=0., maxval=1.)
        differences = G_sample - X
        interpolates = X + (alpha * differences)
        Ig, _ = discriminator(interpolates, Wd, Ws_bs, Ws_nl)
        gradients = tf.gradients(Ig, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    else:
        gradient_penalty = 0

    weight_decay = 0.0001

    DC_loss = - (D_loss + C_loss) + LAMBDA * gradient_penalty + weight_decay * (
                tf.nn.l2_loss(Ws_nl) + tf.nn.l2_loss(Wd)) + tf.nn.l2_loss(Ws_bs - W_sf[:, 0:20000])

    # GAN's G loss
    G_loss = tf.reduce_mean(tf.log(D_fake))  # +weight_decay*(tf.nn.l2_loss(Wz))

    GC_loss = - (G_loss + C_loss)

    # Prediction Accuracy
    P_real = tf.nn.softmax(C_real)
    correct_prediction = tf.equal(tf.argmax(C_real, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    correct_prediction_f = tf.equal(tf.argmax(C_fake, 1), tf.argmax(y, 1))
    accuracy_f = tf.reduce_mean(tf.cast(correct_prediction_f, tf.float32))

    # Gradient Descent
    # Choose learning rate
    learning_rate = cmd.lr
    scale = 100
    if way_opt == 0:
        D_solver = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(DC_loss, var_list=theta_D)
        G_solver = tf.train.MomentumOptimizer(learning_rate * scale, 0.9).minimize(GC_loss, var_list=theta_G)
    elif way_opt == 1:
        D_solver = (tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(DC_loss, var_list=theta_D))
        G_solver = (tf.train.AdamOptimizer(learning_rate=learning_rate * scale).minimize(GC_loss, var_list=theta_G))
    elif way_opt == 2:
        D_solver = (tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(DC_loss, var_list=theta_D))
        G_solver = (tf.train.RMSPropOptimizer(learning_rate=learning_rate * scale).minimize(GC_loss, var_list=theta_G))
    elif way_opt == 3:
        D_solver = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(DC_loss, var_list=theta_D)
        G_solver = tf.train.GradientDescentOptimizer(learning_rate=learning_rate * scale).minimize(GC_loss,
                                                                                                   var_list=theta_G)

    sess = tf.Session()
    # Initializing the variables
    init = tf.global_variables_initializer()

    # Load data file
    pos = 0

    fea_tsv = open(tsvFea, "rb")
    lbl_tsv = open(tsvLbl, "rb")

    # Load validation data and test data
    val_data, val_labels, _, _ = dp.nextbatch(pos, vlSf, len(vlSf), fea_tsv, lbl_tsv, feaIdx, lblIdx)
    test_data, test_labels, _, _ = dp.nextbatch(pos, tsSf, len(tsSf), fea_tsv, lbl_tsv, feaIdx, lblIdx)

    if opt_nml == 1:
        val_data = dp.normalization(val_data)
        test_data = dp.normalization(test_data)

    merged = tf.summary.merge_all()

    fline = G_w[way_Gw], Opt[way_opt], learning_rate, "WGAN:", wgan, "G Scale:", scale
    print(fline)

    with tf.Session() as sess:
        sess.run(init)

        for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
            # Generate Batches
            if pos + BATCH_SIZE >= train_size:
                pos = 0
            X_mb, y_mb, _, _ = dp.nextbatch(pos, trSf, BATCH_SIZE, fea_tsv, lbl_tsv, feaIdx, lblIdx)
            if opt_nml == 1:
                X_mb = dp.normalization(X_mb)

            pos += BATCH_SIZE
            z_mb = sample_z(BATCH_SIZE, z_dim)

            # Optimize Disriminator
            if step % 2 == 0:
                _ = sess.run(D_solver, feed_dict={X: X_mb, y: y_mb, z: z_mb})
            _ = sess.run(G_solver, feed_dict={X: X_mb, y: y_mb, z: z_mb})

            if step % 2000 == 0:
                val_acc, val_pre = sess.run([accuracy, P_real], feed_dict={X: val_data, y: val_labels})
                ts_acc, ts_pre = sess.run([accuracy, P_real], feed_dict={X: test_data, y: test_labels})
                fline2 = 'Iter_acc: {}; Validation Accuracy: {}; Test Accuracy: {}'.format(step, val_acc, ts_acc)
                print(fline2)

    fea_tsv.close()
    lbl_tsv.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', required=True, type=int, help='normalization')
    parser.add_argument('--g', required=True, type=int, help='wgan')
    parser.add_argument('--s', required=True, type=int, help='Wz_scale')
    parser.add_argument('--w', required=True, type=np.float32, help='learning rate')
    parser.add_argument('--lr', required=True, type=np.float32, help='weights')
    parser.add_argument('--i', required=True, type=int, help='inilization')
    return parser.parse_args()


if __name__ == "__main__":
    cmd = parse_args()
    w_gp_ac_gan(cmd)

    ##  python w_gp_ac_gan_face_new2.py --n 1 --g 1 --lr 1e-4 --s 200 --w 0.7 --i 1
