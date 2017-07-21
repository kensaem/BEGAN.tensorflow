from parser import st2list

from layer import *
from selu import *


def leaky_relu(x, neg_thres=0.2):
    y = tf.maximum(x, neg_thres*x)
    return y


def conv_block(
        input_t,
        output_channel,
        layer_name,
        kernel=3,
        stride=1,
        activation=tf.nn.elu,
        padding='SAME'
):
    input_channel = input_t.get_shape().as_list()[-1]
    use_selu = False

    with tf.variable_scope(layer_name):
        if use_selu and activation == tf.nn.elu:
            w_conv = weight_variable_for_selu([kernel, kernel, input_channel, output_channel], name=layer_name)
            b_conv = bias_variable_for_selu([output_channel], name=layer_name)
        else:
            w_conv = weight_variable([kernel, kernel, input_channel, output_channel], name=layer_name)
            b_conv = bias_variable([output_channel], name=layer_name)
        output_t = tf.nn.bias_add(conv2d(input_t, w_conv, stride=[1, stride, stride, 1], padding=padding), b_conv)
        if use_selu and activation == tf.nn.elu:
            output_t = selu(output_t)
        else:
            output_t = activation(output_t)
    return output_t


def deconv_block(
        input_t,
        output_shape,
        layer_name,
        stride=1,
        kernel=3,
        activation=tf.nn.elu,
        padding='SAME'
):
    inpus_channels = input_t.get_shape().as_list()[-1]
    with tf.variable_scope(layer_name):
        w_conv = weight_variable([kernel, kernel, output_shape[-1], inpus_channels], name=layer_name)
        output_t = tf.nn.conv2d_transpose(input_t, w_conv, output_shape, [1, stride, stride, 1], padding=padding)

        b_conv = bias_variable([output_shape[-1]], name=layer_name)
        output_t = tf.nn.bias_add(output_t, b_conv)

        output_t = activation(output_t)
    return output_t


class BEGANModel:
    def __init__(self, noise_size, channel_size, batch_size=16):

        self.noise_size = noise_size
        self.n_ch = channel_size
        self.batch_size = batch_size

        self.image_info = {
            'w': 128,
            'h': 128,
            'c': 3,
        }
        self.scale_factor = 4  # width_height / (2^scale_factor) = 8

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.prop_k = tf.Variable(0.0, trainable=False, name='proportional_control_theory_k')

        # NOTE placeholder for hyper-parameters of learning
        self.lr_ph = tf.placeholder(dtype=tf.float32, name='learning_rate')

        self.lambda_k_ph = tf.placeholder(dtype=tf.float32, name='lambda_k')

        self.gamma_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(),
            name='gamma_placeholder'
        )

        # NOTE placeholder for model
        self.batch_size_ph = tf.placeholder(
            dtype=tf.int32,
            shape=(),
            name='batch_size_placeholder'
        )

        self.input_image_ph = tf.placeholder(
            dtype=tf.uint8,
            shape=[None, self.image_info['w'], self.image_info['h'], self.image_info['c']],
            name='input_image_placeholder')

        self.noise_ph = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.noise_size],
            name='noise_placeholder',
        )

        # NOTE Build model for generator
        self.z_t = tf.random_uniform(shape=[self.batch_size_ph, self.noise_size], minval=-1.0, maxval=1.0)

        self.fake_image_t = self.build_gen_decoder(
            input_t=self.z_t,
            n_ch=self.n_ch,
            h_size=self.noise_size,
            name="generator"
        )

        # NOTE Build model for discriminator
        self.real_image_t = tf.div(tf.to_float(self.input_image_ph), 127.5) - 1.0

        self.ae_fake_t = self.build_auto_encoder(input_t=self.fake_image_t, n_ch=self.n_ch, h_size=self.noise_size, name="auto_encoder")
        self.ae_real_t = self.build_auto_encoder(input_t=self.real_image_t, n_ch=self.n_ch, h_size=self.noise_size, name="auto_encoder", reuse=True)

        # NOTE Discriminator loss
        self.ae_loss_real = tf.reduce_mean(tf.abs(self.ae_real_t - self.real_image_t))
        self.ae_loss_fake = tf.reduce_mean(tf.abs(self.ae_fake_t - self.fake_image_t))
        self.disc_loss = self.ae_loss_real - self.prop_k * self.ae_loss_fake

        # NOTE Generator loss
        self.gen_loss = self.ae_loss_fake

        self.balance = self.gamma_ph * self.ae_loss_real - self.gen_loss
        self.measure = self.ae_loss_real + tf.abs(self.balance)

        # NOTE build optimizers
        optimizer = tf.train.AdamOptimizer

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if "auto_encoder" in var.name]
        print("\n==== Variables for auto_encoder(discriminator) ====")
        for var in d_vars:
            print(var.name)
        self.train_op_disc = optimizer(self.lr_ph).minimize(
            loss=self.disc_loss,
            var_list=d_vars,
        )

        g_vars = [var for var in t_vars if "generator" in var.name]
        print("\n==== Variables for generator ====")
        for var in g_vars:
            print(var.name)
        self.train_op_gen = optimizer(self.lr_ph).minimize(
            loss=self.gen_loss,
            global_step=self.global_step,
            var_list=g_vars,
        )

        # NOTE Update k
        with tf.control_dependencies([self.train_op_disc, self.train_op_gen]):
            self.update_prop_k = tf.assign(
                self.prop_k,
                tf.clip_by_value(self.prop_k + self.lambda_k_ph * self.balance, 0.0, 1.0)
            )

        tf.summary.histogram("real_image", self.real_image_t)
        tf.summary.histogram("fake_image", self.fake_image_t)
        tf.summary.scalar("balance", self.balance)
        tf.summary.scalar("measure", self.measure)
        tf.summary.scalar("disc_loss", self.disc_loss)
        tf.summary.scalar("gen_loss", self.gen_loss)
        tf.summary.scalar("prop_k", self.prop_k)
        tf.summary.scalar("lr", self.lr_ph)

        return

    def build_gen_decoder(
            self,
            input_t,
            n_ch,
            h_size,
            reuse=False,
            name="generator_or_decoder",
    ):
        repeat_n = self.scale_factor + 1

        with tf.variable_scope(name, reuse=reuse):
            with tf.variable_scope("preprocessing"):
                output_t = input_t

            with tf.variable_scope("fc_first"):
                w_fc = weight_variable([h_size, 8*8*n_ch], name="fc_first")
                b_fc = bias_variable([8*8*n_ch], name="fc_first")
                output_t = tf.nn.bias_add(tf.matmul(output_t, w_fc), b_fc)
                # output_t = tf.nn.elu(output_t)

            output_t = tf.reshape(output_t, [-1, 8, 8, n_ch])
            h0 = output_t

            for i in range(repeat_n):
                output_t = conv_block(
                    output_t,
                    n_ch,
                    layer_name="conv_"+str(i+1)+"_1",
                )
                output_t = conv_block(
                    output_t,
                    n_ch,
                    layer_name="conv_"+str(i+1)+"_2",
                )

                if i != (repeat_n-1):
                    w_h = 8 * 2**(i+1)
                    output_t = tf.concat(
                        [tf.image.resize_nearest_neighbor(output_t, [w_h, w_h]),
                         tf.image.resize_nearest_neighbor(h0, [w_h, w_h])],
                        axis=3
                    )

            output_t = conv_block(
                output_t,
                3,
                layer_name="conv_last",
                # activation=tf.nn.tanh
            )

        return output_t

    def build_encoder(
            self,
            input_t,
            n_ch,
            h_size,
            reuse=False,
            name="encoder"
    ):

        output_t = input_t
        repeat_n = self.scale_factor+1

        with tf.variable_scope(name, reuse=reuse):

            output_t = conv_block(
                output_t,
                n_ch,
                layer_name="conv_first",
            )

            for i in range(repeat_n):
                output_t = conv_block(
                    output_t,
                    (i+1)*n_ch,
                    layer_name="conv_"+str(i+1)+"_1",
                )

                if i != (repeat_n-1):
                    n_ch_factor = i+2
                    stride = 2
                else:
                    n_ch_factor = i+1
                    stride = 1

                output_t = conv_block(
                    output_t,
                    n_ch_factor*n_ch,
                    layer_name="conv_"+str(i+1)+"_2",
                    stride=stride,
                    )

            print(output_t)

            # input size 2 w/ channel 512 => fc layer
            output_t = tf.reshape(output_t, [-1, 8*8*repeat_n*n_ch])

            with tf.variable_scope("disc_fc"):
                w_conv = weight_variable([8*8*repeat_n*n_ch, h_size], name="disc_fc")
                b_conv = bias_variable([h_size], name="disc_fc")
                output_t = tf.nn.bias_add(tf.matmul(output_t, w_conv), b_conv)

            print(output_t)

        return output_t

    def build_auto_encoder(
            self,
            input_t,
            n_ch,
            h_size,
            reuse=False,
            name="auto_encoder"):

        output_t = input_t

        with tf.variable_scope(name, reuse=reuse):
            output_t = self.build_encoder(
                input_t=output_t,
                n_ch=n_ch,
                h_size=h_size,
                reuse=reuse,
                name="encoder"
            )

            output_t = self.build_gen_decoder(
                input_t=output_t,
                n_ch=n_ch,
                h_size=h_size,
                reuse=reuse,
                name="decoder"
            )

        return output_t

