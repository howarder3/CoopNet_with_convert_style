from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import myops as ops
import myops2 as ops2
import time
import tensorflow as tf

from six.moves import xrange

from model.utils.interpolate import *
from model.utils.custom_ops import *
from model.utils.data_io import DataSet, saveSampleResults
from model.utils.vis_util import Visualizer


class CoopNets(object):
    def __init__(self, num_epochs=200, image_size=64, batch_size=100, nTileRow=12, nTileCol=12, net_type='object',
                 d_lr=0.001, g_lr=0.0001, beta1=0.5,
                 des_step_size=0.002, des_sample_steps=10, des_refsig=0.016,
                 gen_step_size=0.1, gen_sample_steps=0, gen_refsig=0.3,
                 data_path='/tmp/data/', log_step=10, 
                 category='orange_256',
                 category2='apple_256',
                 category3='test_64',
                 sample_dir='./synthesis', model_dir='./checkpoints', log_dir='./log', test_dir='./test'):
        self.type = net_type
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.nTileRow = nTileRow
        self.nTileCol = nTileCol
        self.num_chain = nTileRow * nTileCol
        self.beta1 = beta1


        self.d_lr = d_lr
        self.g_lr = g_lr
        self.delta1 = des_step_size
        self.sigma1 = des_refsig
        self.delta2 = gen_step_size
        self.sigma2 = gen_refsig
        self.t1 = des_sample_steps
        self.t2 = gen_sample_steps

        self.data_path = os.path.join(data_path, category)
        self.data_path2 = os.path.join(data_path, category2)
        self.data_path3 = os.path.join(data_path, category3)
        self.log_step = log_step

        self.log_dir = log_dir
        self.sample_dir = sample_dir
        self.model_dir = model_dir
        self.test_dir = test_dir

        if self.type == 'texture':
            self.z_size = 49
        elif self.type == 'object':
            self.z_size = 100
        elif self.type == 'object_small':
            self.z_size = 2

        self.syn = tf.placeholder(shape=[None, self.image_size, self.image_size, 3], dtype=tf.float32, name='syn')
        self.obs = tf.placeholder(shape=[None, self.image_size, self.image_size, 3], dtype=tf.float32, name='obs')
        # self.z = tf.placeholder(shape=[None, self.z_size], dtype=tf.float32, name='z')
        self.z = tf.placeholder(shape=[None, self.image_size, self.image_size, 3], dtype=tf.float32, name='z')

        self.debug = False
        
        #  --- adding new parameters --- 
        self.ngf = 64
        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')
        self.norm='instance'
        self.reuse = False


    def build_model(self):
        self.gen_res = self.generator(self.z, reuse=False)

        obs_res = self.descriptor(self.obs, reuse=False)
        syn_res = self.descriptor(self.syn, reuse=True)

        self.recon_err = tf.reduce_mean(
            tf.pow(tf.subtract(tf.reduce_mean(self.syn, axis=0), tf.reduce_mean(self.obs, axis=0)), 2))

        # descriptor variables
        des_vars = [var for var in tf.trainable_variables() if var.name.startswith('des')]

        self.des_loss = tf.reduce_sum(tf.subtract(tf.reduce_mean(syn_res, axis=0), tf.reduce_mean(obs_res, axis=0)))

        des_optim = tf.train.AdamOptimizer(self.d_lr, beta1=self.beta1)
        des_grads_vars = des_optim.compute_gradients(self.des_loss, var_list=des_vars)
        # des_grads = [tf.reduce_mean(tf.abs(grad)) for (grad, var) in des_grads_vars if '/w' in var.name]
        # update by mean of gradients
        self.apply_d_grads = des_optim.apply_gradients(des_grads_vars)

        # generator variables
        gen_vars = [var for var in tf.trainable_variables() if var.name.startswith('gen')]

        self.gen_loss = tf.reduce_sum(tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.obs - self.gen_res),
                                       axis=0))

        gen_optim = tf.train.AdamOptimizer(self.g_lr, beta1=self.beta1)
        gen_grads_vars = gen_optim.compute_gradients(self.gen_loss, var_list=gen_vars)
        # gen_grads = [tf.reduce_mean(tf.abs(grad)) for (grad, var) in gen_grads_vars if '/w' in var.name]
        self.apply_g_grads = gen_optim.apply_gradients(gen_grads_vars)

        # symbolic langevins
        self.langevin_descriptor = self.langevin_dynamics_descriptor(self.syn)
        self.langevin_generator = self.langevin_dynamics_generator(self.z)

        tf.summary.scalar('des_loss', self.des_loss)
        tf.summary.scalar('gen_loss', self.gen_loss)
        tf.summary.scalar('recon_err', self.recon_err)

        self.summary_op = tf.summary.merge_all()

    def langevin_dynamics_descriptor(self, syn_arg):
        def cond(i, syn):
            return tf.less(i, self.t1)

        def body(i, syn):
            noise = tf.random_normal(shape=[self.num_chain, self.image_size, self.image_size, 3], name='noise')
            syn_res = self.descriptor(syn, reuse=True)
            grad = tf.gradients(syn_res, syn, name='grad_des')[0]
            syn = syn - 0.5 * self.delta1 * self.delta1 * (syn / self.sigma1 / self.sigma1 - grad) + self.delta1 * noise
            return tf.add(i, 1), syn

        with tf.name_scope("langevin_dynamics_descriptor"):
            i = tf.constant(0)
            i, syn = tf.while_loop(cond, body, [i, syn_arg])
            return syn

    def langevin_dynamics_generator(self, z_arg):
        def cond(i, z):
            return tf.less(i, self.t2)

        def body(i, z):
            # noise = tf.random_normal(shape=[self.num_chain, self.z_size], name='noise')
            noise = tf.random_normal(shape=[self.ngf, self.ngf, 3], name='noise')
            gen_res = self.generator(z, reuse=True)
            gen_loss = tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.obs - gen_res),
                                       axis=0)

            grad = tf.gradients(gen_loss, z, name='grad_gen')[0]
            # print('z.shape = {}'.format(z.shape)) # z.shape = (?, 64, 64, 3)
            # print('grad.shape = {}'.format(grad.shape)) # grad.shape = (?, 64, 64, 3)
            # print('self.delta2 = {}'.format(self.delta2)) # self.delta2 = 0.1
            # print('noise.shape = {}'.format(noise.shape)) # z.shape = (?, 64, 64, 3)
            # z = z - 0.5 * self.delta2 * self.delta2 * (z + grad) + self.delta2 * noise
            z = z - 0.5 * self.delta2 * self.delta2 * (z + grad) * noise

            return tf.add(i, 1), z

        with tf.name_scope("langevin_dynamics_generator"):
            i = tf.constant(0)
            i, z = tf.while_loop(cond, body, [i, z_arg])
            return z

    def train(self, sess):
        self.build_model()

        # Prepare training data
        train_data = DataSet(self.data_path, image_size=self.image_size)
        # print('train_data.shape  = {}'.format(train_data.shape))
        origin_data = DataSet(self.data_path2, image_size=self.image_size)
        # print('origin_data.shape  = {}'.format(origin_data.shape))

        # num_batches = int(math.ceil(len(train_data) / self.batch_size))

        num_batches = 10

        # print('batch_size  = {}'.format(self.batch_size)) # batch_size  = 100
        # print('num_batches = {}'.format(num_batches)) # num_batches = 12

        # initialize training
        sess.run(tf.global_variables_initializer())

        sample_results = np.random.randn(self.num_chain * num_batches, self.image_size, self.image_size, 3)
        # print('self.num_chain = {}'.format(self.num_chain)) # self.num_chain = 144
        # print('sample_results.shape = {}'.format(sample_results.shape)) # sample_results.shape = (1728, 64, 64, 3)
        saver = tf.train.Saver(max_to_keep=50)

        # make graph immutable
        tf.get_default_graph().finalize()

        # store graph in protobuf
        with open(self.model_dir + '/graph.proto', 'w') as f:
            f.write(str(tf.get_default_graph().as_graph_def()))

        des_loss_vis = Visualizer(title='descriptor', ylabel='normalized negative log-likelihood', ylim=(-200, 200),
                                  save_figpath=self.log_dir + '/des_loss.png', avg_period = self.batch_size)

        gen_loss_vis = Visualizer(title='generator', ylabel='reconstruction error',
                                  save_figpath=self.log_dir + '/gen_loss.png', avg_period = self.batch_size)


        # train
        for epoch in xrange(self.num_epochs):
            start_time = time.time()
            des_loss_avg, gen_loss_avg, mse_avg = [], [], []
            for i in xrange(num_batches):

                obs_data = train_data[i * self.batch_size:min(len(train_data), (i + 1) * self.batch_size)]
                # print('obs_data.shape = {}'.format(obs_data.shape)) # obs_data.shape = (100, 64, 64, 3)
                
                ori_data = origin_data[i * self.num_chain:min(len(origin_data), (i + 1) * self.num_chain)]
                # print('(i + 1) = {}'.format((i + 1)))
                # print('self.num_chain = {}'.format(self.num_chain)) # self.num_chain = 100
                # print('len(origin_data) = {}'.format(len(origin_data))) # len(origin_data) = 1504
                # print('ori_data.shape = {}'.format(ori_data.shape)) # origin_data.shape = (100, 64, 64, 3)

                # ori_data = np.reshape(ori_data, (100, 64*64*3)) 
                # print('ori_data.shape = {}'.format(ori_data.shape)) # origin_data.shape = (100, 12288)
                # Step G0: generate X ~ N(0, 1)
                
                '''
                z_vec = np.random.randn(self.num_chain, self.z_size) # z_vec.shape = (144, 100) 
                print('G0 : self.num_chain = {}'.format(self.num_chain)) # self.num_chain = 144
                print('G0 : self.z_size = {}'.format(self.z_size)) # self.z_size = 100
                print('G0 : z_vec.shape = {}'.format(z_vec.shape)) # z_vec.shape = (144, 100)
                '''

                z_vec = ori_data
                # print('G0 : z_vec.shape = {}'.format(z_vec.shape)) # z_vec.shape = (100, 12288)
               

                # print('G0 : z_vec = {}'.format(z_vec))
                # print(type(z_vec))
                # print('G0 : z_vec = {}'.format(z_vec.tolist()))
                '''
                z_vec_list = z_vec.tolist()
                print(len(z_vec_list))  
                for ele in (z_vec_list): #144
                    # print(len(ele))
                    pass
                    # print()
                    for ele2 in ele: #100
                        pass
                        # print("{:.1f}".format(ele2),end=",")
                '''

                g_res = sess.run(self.gen_res, feed_dict={self.z: z_vec})
                # print('G0 : g_res.shape = {}'.format(g_res.shape)) # g_res.shape = (144, 64, 64, 3)
                # Step D1: obtain synthesized images Y
                if self.t1 > 0:
                    syn = sess.run(self.langevin_descriptor, feed_dict={self.syn: g_res})
                    # print(' D1 : syn.shape = {}'.format(syn.shape)) # syn.shape = (144, 64, 64, 3)
                # Step G1: update X using Y as training image
                if self.t2 > 0:
                    z_vec = sess.run(self.langevin_generator, feed_dict={self.z: z_vec, self.obs: syn})
                    # print(' G1 : z_vec.shape = {}'.format(z_vec.shape))
                # Step D2: update D net
                d_loss = sess.run([self.des_loss, self.apply_d_grads],
                                  feed_dict={self.obs: obs_data, self.syn: syn})[0]
                # print('  D2 : d_loss = {}'.format(d_loss))
                # Step G2: update G net
                g_loss = sess.run([self.gen_loss, self.apply_g_grads],
                                  feed_dict={self.obs: syn, self.z: z_vec})[0]
                # print('  G2 : g_loss = {}'.format(g_loss))

                # Compute MSE for generator
                mse = sess.run(self.recon_err, feed_dict={self.obs: syn, self.syn: g_res})
                sample_results[i * self.num_chain:(i + 1) * self.num_chain] = syn

                des_loss_avg.append(d_loss)
                gen_loss_avg.append(g_loss)
                mse_avg.append(mse)

                des_loss_vis.add_loss_val(epoch*num_batches + i, d_loss / float(self.image_size * self.image_size * 3))
                gen_loss_vis.add_loss_val(epoch*num_batches + i, mse)

                if self.debug:
                    print('Epoch #{:d}, [{:2d}]/[{:2d}], descriptor loss: {:.4f}, generator loss: {:.4f}, '
                          'L2 distance: {:4.4f}'.format(epoch, i + 1, num_batches, d_loss.mean(), g_loss.mean(), mse))
                if i == 0 and epoch % self.log_step == 0:
                    if not os.path.exists(self.sample_dir):
                        os.makedirs(self.sample_dir)
                    saveSampleResults(syn, "%s/des%03d.png" % (self.sample_dir, epoch), col_num=self.nTileCol)
                    saveSampleResults(g_res, "%s/gen%03d.png" % (self.sample_dir, epoch), col_num=self.nTileCol)

            end_time = time.time()
            print('Epoch #{:d}, avg.descriptor loss: {:.4f}, avg.generator loss: {:.4f}, avg.L2 distance: {:4.4f}, '
                  'time: {:.2f}s'.format(epoch, np.mean(des_loss_avg), np.mean(gen_loss_avg), np.mean(mse_avg), end_time - start_time))

            if epoch % self.log_step == 0:
                if not os.path.exists(self.model_dir):
                    os.makedirs(self.model_dir)
                saver.save(sess, "%s/%s" % (self.model_dir, 'model.ckpt'), global_step=epoch)

                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)

                des_loss_vis.draw_figure()
                gen_loss_vis.draw_figure()


    def test(self, sess, ckpt, sample_size):
        assert ckpt is not None, 'no checkpoint provided.'

        # gen_res = self.generator(self.z, reuse=False)
        self.gen_res = self.generator(self.z, reuse=False)

        # num_batches = int(math.ceil(sample_size / self.num_chain))
        num_batches = 100

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)
        print('Loading checkpoint {}.'.format(ckpt))

        test_data = DataSet(self.data_path3, image_size=self.image_size)

        for i in xrange(num_batches):
            # z_vec = np.random.randn(min(sample_size, self.num_chain), self.z_size)
            test_data_batch = test_data[i * self.batch_size:min(len(test_data), (i + 1) * self.batch_size)]
            # print('(i + 1) = {}'.format((i + 1)))
            # print('self.batch_size = {}'.format(self.batch_size)) # self.batch_size = 100
            # print('len(origin_data) = {}'.format(len(origin_data))) # len(origin_data) = 1504
            # print('ori_data.shape = {}'.format(ori_data.shape)) # origin_data.shape = (100, 64, 64, 3)

            # ori_data = np.reshape(ori_data, (100, 64*64*3)) 
            # print('ori_data.shape = {}'.format(ori_data.shape)) # origin_data.shape = (100, 12288)
            # Step G0: generate X ~ N(0, 1)
            
            '''
            z_vec = np.random.randn(self.num_chain, self.z_size) # z_vec.shape = (144, 100) 
            print('G0 : self.num_chain = {}'.format(self.num_chain)) # self.num_chain = 144
            print('G0 : self.z_size = {}'.format(self.z_size)) # self.z_size = 100
            print('G0 : z_vec.shape = {}'.format(z_vec.shape)) # z_vec.shape = (144, 100)
            '''

            z_vec = test_data_batch


            g_res = sess.run(self.gen_res, feed_dict={self.z: z_vec})
            # g_res = sess.run(self.gen_res, feed_dict={self.z: z_vec})
            saveSampleResults(g_res, "%s/gen%03d.png" % (self.test_dir, i), col_num=self.nTileCol)

            # # output interpolation results
            # interp_z = linear_interpolator(z_vec, npairs=self.nTileRow, ninterp=self.nTileCol)
            # interp = sess.run(gen_res, feed_dict={self.z: interp_z})
            # saveSampleResults(interp, "%s/interp%03d.png" % (self.test_dir, i), col_num=self.nTileCol)
            # sample_size = sample_size - self.num_chain




    def descriptor(self, inputs, reuse=False):
        '''
        conv1.shape = (?, 32, 32, 64)
        conv2.shape = (?, 16, 16, 128)
        conv3.shape = (?, 16, 16, 256)
        fc.shape = (?, 1, 1, 100)
        '''

        # with tf.variable_scope('des', reuse=reuse):
        #     # conv layers
        #     print('des_inputs.shape = {}'.format(inputs.shape)) # (?, 64, 64, 3)
        #     c7s1_32 = ops2.c7s1_k(inputs, self.ngf, is_training=self.is_training, norm=self.norm,
        #     reuse=self.des_reuse, name='c7s1_32')                            
        #     print('des_c7s1_32.shape = {}'.format(c7s1_32.shape)) # (?, 64, 64, 64)
        #     d64 = ops2.dk(c7s1_32, 2*self.ngf, is_training=self.is_training, norm=self.norm,
        #     reuse=self.des_reuse, name='d64')                                
        #     print('des_d64.shape = {}'.format(d64.shape)) # (?, 32, 32, 128)
        #     d128 = ops2.dk(d64, 4*self.ngf, is_training=self.is_training, norm=self.norm,
        #     reuse=self.des_reuse, name='d128')                               
        #     print('des_d128.shape = {}'.format(d128.shape)) # (?, 16, 16, 256)

        #     if self.image_size <= 128:
        #         # use 6 residual blocks for 128x128 images
        #         res_output = ops2.n_res_blocks(d128, reuse=self.des_reuse, n=6)      
        #     else:
        #         # 9 blocks for higher resolution
        #         res_output = ops2.n_res_blocks(d128, reuse=self.des_reuse, n=9)      



        #     print('des_res_output.shape = {}'.format(res_output.shape))  # res_output.shape = (?, 16, 16, 256)
        #     convt1 = convt2d(res_output, (None, self.image_size // 2, self.image_size // 2, 2*self.ngf), kernal=(5, 5)
        #      , strides=(2, 2), padding="SAME", name="convt1")
        #     convt1 = tf.contrib.layers.batch_norm(convt1, is_training=self.is_training)
        #     convt1 = leaky_relu(convt1)
        #     print('des_convt1.shape = {}'.format(convt1.shape)) # convt1.shape = (?, 4, 4, 512) -> (?, 32, 32, 128) 

        #     convt2 = convt2d(convt1, (None, self.image_size , self.image_size, self.ngf), kernal=(5, 5)
        #      , strides=(2, 2), padding="SAME", name="convt2")
        #     convt2 = tf.contrib.layers.batch_norm(convt2, is_training=self.is_training)
        #     convt2 = leaky_relu(convt2)
        #     print('des_convt2.shape = {}'.format(convt2.shape)) # convt1.shape = (?, 64, 64, 64)

        #     # conv layer
        #     # Note: the paper said that ReLU and _norm were used
        #     # but actually tanh was used and no _norm here
        #     output = ops2.c7s1_k(convt2, 3, norm=None, activation='tanh', reuse=self.des_reuse, name='output') 
        #     print('des_output.shape = {}'.format(output.shape)) # output.shape = (?, 64, 64, 64)


# def c7s1_k(input, k, reuse=False, norm='instance', activation='relu', is_training=True, name='c7s1_k'):
#   """ A 7x7 Convolution-BatchNorm-ReLU layer with k filters and stride 1
#   Args:
#     input: 4D tensor
#     k: integer, number of filters (output depth)
#     norm: 'instance' or 'batch' or None
#     activation: 'relu' or 'tanh'
#     name: string, e.g. 'c7sk-32'
#     is_training: boolean or BoolTensor
#     name: string
#     reuse: boolean
#   Returns:
#     4D tensor

#     conv1 = conv2d(inputs, 64, kernal=(5, 5), strides=(2, 2), padding="SAME", activate_fn=leaky_relu,
#                                name="conv1")
#   """
#   with tf.variable_scope(name, reuse=reuse):
#     weights = _weights("weights",
#       shape=[7, 7, input.get_shape()[3], k])

#     padded = tf.pad(input, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
#     conv = tf.nn.conv2d(padded, weights, strides=[1, 1, 1, 1], padding='VALID')

#     normalized = _norm(conv, is_training, norm)

#     if activation == 'relu':
#       output = tf.nn.relu(normalized)
#     if activation == 'tanh':
#       output = tf.nn.tanh(normalized)
#     return output
#             '''

            # set reuse=True for next call
            # self.des_reuse = True
            # # self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            # return output
        
        with tf.variable_scope('des', reuse=reuse):
            if self.type == 'object':
                conv1 = conv2d(inputs, 64, kernal=(5, 5), strides=(2, 2), padding="SAME", activate_fn=leaky_relu,
                               name="conv1")
                print('conv1.shape = {}'.format(conv1.shape))

                conv2 = conv2d(conv1, 128, kernal=(3, 3), strides=(2, 2), padding="SAME", activate_fn=leaky_relu,
                               name="conv2")
                print('conv2.shape = {}'.format(conv2.shape))

                conv3 = conv2d(conv2, 256, kernal=(3, 3), strides=(1, 1), padding="SAME", activate_fn=leaky_relu,
                               name="conv3")
                print('conv3.shape = {}'.format(conv3.shape))

                fc = fully_connected(conv3, 100, name="fc")
                print('fc.shape = {}'.format(fc.shape))

                return fc
            else:
                return NotImplementedError

    def generator(self, inputs, reuse=False, is_training=True):
        '''
        inputs.shape = (?, 1, 1, 100)
        convt1.shape = (?, 4, 4, 512)
        convt2.shape = (?, 8, 8, 256)
        convt3.shape = (?, 16, 16, 128)
        convt4.shape = (?, 32, 32, 64)
        convt5.shape = (?, 64, 64, 3)
        '''
        '''
        inputs.shape = (?, 64, 64, 3)
        c7s1_32.shape = (?, 64, 64, 64)
        d64.shape = (?, 32, 32, 128)
        d128.shape = (?, 16, 16, 256)
        '''
        with tf.variable_scope('gen', reuse=reuse):
            # conv layers
            print('inputs.shape = {}'.format(inputs.shape)) # (?, 64, 64, 3)
            c7s1_32 = ops.c7s1_k(inputs, self.ngf, is_training=self.is_training, norm=self.norm,
            reuse=self.reuse, name='c7s1_32')                            
            print('c7s1_32.shape = {}'.format(c7s1_32.shape)) # (?, 64, 64, 64)
            d64 = ops.dk(c7s1_32, 2*self.ngf, is_training=self.is_training, norm=self.norm,
            reuse=self.reuse, name='d64')                                
            print('d64.shape = {}'.format(d64.shape)) # (?, 32, 32, 128)
            d128 = ops.dk(d64, 4*self.ngf, is_training=self.is_training, norm=self.norm,
            reuse=self.reuse, name='d128')                               
            print('d128.shape = {}'.format(d128.shape)) # (?, 16, 16, 256)

            if self.image_size <= 128:
                # use 6 residual blocks for 128x128 images
                res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=6)      
            else:
                # 9 blocks for higher resolution
                res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=9)      



            print('res_output.shape = {}'.format(res_output.shape))  # res_output.shape = (?, 16, 16, 256)
            convt1 = convt2d(res_output, (None, self.image_size // 2, self.image_size // 2, 2*self.ngf), kernal=(5, 5)
             , strides=(2, 2), padding="SAME", name="convt1")
            convt1 = tf.contrib.layers.batch_norm(convt1, is_training=is_training)
            convt1 = leaky_relu(convt1)
            print('convt1.shape = {}'.format(convt1.shape)) # convt1.shape = (?, 4, 4, 512) -> (?, 32, 32, 128) 

            convt2 = convt2d(convt1, (None, self.image_size , self.image_size, self.ngf), kernal=(5, 5)
             , strides=(2, 2), padding="SAME", name="convt2")
            convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
            convt2 = leaky_relu(convt2)
            print('convt2.shape = {}'.format(convt2.shape)) # convt1.shape = (?, 64, 64, 64)

            # conv layer
            # Note: the paper said that ReLU and _norm were used
            # but actually tanh was used and no _norm here
            output = ops.c7s1_k(convt2, 3, norm=None, activation='tanh', reuse=self.reuse, name='output') 
            print('output.shape = {}'.format(output.shape)) # output.shape = (?, 64, 64, 64)

            '''
def c7s1_k(input, k, reuse=False, norm='instance', activation='relu', is_training=True, name='c7s1_k'):
  """ A 7x7 Convolution-BatchNorm-ReLU layer with k filters and stride 1
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    norm: 'instance' or 'batch' or None
    activation: 'relu' or 'tanh'
    name: string, e.g. 'c7sk-32'
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
  Returns:
    4D tensor

    conv1 = conv2d(inputs, 64, kernal=(5, 5), strides=(2, 2), padding="SAME", activate_fn=leaky_relu,
                               name="conv1")
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[7, 7, input.get_shape()[3], k])

    padded = tf.pad(input, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
    conv = tf.nn.conv2d(padded, weights, strides=[1, 1, 1, 1], padding='VALID')

    normalized = _norm(conv, is_training, norm)

    if activation == 'relu':
      output = tf.nn.relu(normalized)
    if activation == 'tanh':
      output = tf.nn.tanh(normalized)
    return output
            '''

            # set reuse=True for next call
            self.gen_reuse = True
            # self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output

'''
convt5 = convt2d(convt2, (None, self.image_size, self.image_size, 3), kernal=(5, 5)
         , strides=(2, 2), padding="SAME", name="convt5")
convt5 = tf.nn.tanh(convt5)
print('convt5.shape = {}'.format(convt5.shape)) # convt5.shape = (?, 64, 64, 3)

return convt5
'''


'''
# fractional-strided convolution

u64 = ops.uk(res_output, 2*self.ngf, is_training=self.is_training, norm=self.norm,
reuse=self.reuse, name='u64')                                 # (?, w/2, h/2, 64)

u32 = ops.uk(u64, self.ngf, is_training=self.is_training, norm=self.norm,
reuse=self.reuse, name='u32', output_size=self.image_size)         # (?, w, h, 32)

# conv layer
# Note: the paper said that ReLU and _norm were used
# but actually tanh was used and no _norm here
output = ops.c7s1_k(u32, 3, norm=None,
activation='tanh', reuse=self.reuse, name='output')           # (?, w, h, 3)

print('input.shape = {}'.format(input.shape))
print('c7s1_32.shape = {}'.format(c7s1_32.shape))
print('d64.shape = {}'.format(d64.shape))
print('d128.shape = {}'.format(d128.shape))
print('res_output.shape = {}'.format(res_output.shape))
print('u64.shape = {}'.format(u64.shape))
print('u32.shape = {}'.format(u32.shape))
print('output.shape = {}'.format(output.shape))


# set reuse=True for next call
self.reuse = True
self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

return output

if self.image_size <= 128:
    # use 6 residual blocks for 128x128 images
    res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=6)      # (?, w/4, h/4, 128)
else:
    # 9 blocks for higher resolution
    res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=9)      # (?, w/4, h/4, 128)


# fractional-strided convolution
print('res_output.shape = {}'.format(res_output.shape))
u64 = ops.uk(res_output, 2*self.ngf, is_training=self.is_training, norm=self.norm,
reuse=self.reuse, name='u64')                                 # (?, w/2, h/2, 64)
print('u64.shape = {}'.format(u64.shape))
u32 = ops.uk(u64, self.ngf, is_training=self.is_training, norm=self.norm,
reuse=self.reuse, name='u32', output_size=self.image_size)         # (?, w, h, 32)
print('u32.shape = {}'.format(u32.shape))

# conv layer
# Note: the paper said that ReLU and _norm were used
# but actually tanh was used and no _norm here
out = ops.c7s1_k(u32, 3, norm=None, activation='tanh', reuse=self.reuse, name='output')           # (?, w, h, 3)
print('output.shape = {}'.format(output.shape))

# # set reuse=True for next call
self.reuse = True
self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

return out
'''
            


"""
if self.type == 'object':
print('inputs.shape = {}'.format(inputs.shape))
inputs = tf.reshape(inputs, [-1, 1, 1, self.z_size])
print('inputs.shape = {}'.format(inputs.shape))
convt1 = convt2d(inputs, (None, self.image_size // 16, self.image_size // 16, 512), kernal=(4, 4)
             , strides=(1, 1), padding="VALID", name="convt1")
convt1 = tf.contrib.layers.batch_norm(convt1, is_training=is_training)
convt1 = leaky_relu(convt1)
print('convt1.shape = {}'.format(convt1.shape))

convt2 = convt2d(convt1, (None, self.image_size // 8, self.image_size // 8, 256), kernal=(5, 5)
             , strides=(2, 2), padding="SAME", name="convt2")
convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
convt2 = leaky_relu(convt2)
print('convt2.shape = {}'.format(convt2.shape))

convt3 = convt2d(convt2, (None, self.image_size // 4, self.image_size // 4, 128), kernal=(5, 5)
             , strides=(2, 2), padding="SAME", name="convt3")
convt3 = tf.contrib.layers.batch_norm(convt3, is_training=is_training)
convt3 = leaky_relu(convt3)
print('convt3.shape = {}'.format(convt3.shape))

convt4 = convt2d(convt3, (None, self.image_size // 2, self.image_size // 2, 64), kernal=(5, 5)
             , strides=(2, 2), padding="SAME", name="convt4")
convt4 = tf.contrib.layers.batch_norm(convt4, is_training=is_training)
convt4 = leaky_relu(convt4)
print('convt4.shape = {}'.format(convt4.shape))

convt5 = convt2d(convt4, (None, self.image_size, self.image_size, 3), kernal=(5, 5)
             , strides=(2, 2), padding="SAME", name="convt5")
convt5 = tf.nn.tanh(convt5)
print('convt5.shape = {}'.format(convt5.shape))

return convt5
else:
return NotImplementedError
"""
