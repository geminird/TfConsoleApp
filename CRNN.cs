using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using static Tensorflow.Binding;
using Tensorflow.Contrib;
using NumSharp;
using Tensorflow.Operations;
using Tensorflow.Operations.Activation;
using System.Collections.Generic;

namespace TfConsoleApp
{
    public class CRNN
    {
        private int inputHeight;
        private int inputWidth;
        private int batchSize;
        private bool pretrain;

        bool trainable = false;
        Tensor inputs = null;
        Tensor seqLength = null;
        int classNum = 0;
        public CRNN(Tensor inputs, Tensor seqLen, int batch_size, int inputImgHeight, int inputImgWidth, int classNum,
            bool trainable = false)
        {
            this.inputHeight = inputImgHeight;
            this.inputWidth = inputImgWidth;
            this.classNum = classNum;
            this.inputs = inputs;
            this.batchSize = batch_size;
            this.seqLength = seqLen;
            this.trainable = trainable;
        }

        public (Tensor, Tensor[], int) ConstructGraph()
        {
            //进入cnn网络层 shape [batch, length, 32 ,1]
            var cnnOut = this.cnn(this.inputs);
            //送入rnn前将cnn进行reshape
            var maxCharCount = cnnOut.shape[1];
            System.Console.WriteLine(maxCharCount);
            var crnnMode = this._rnn(cnnOut, this.seqLength);
            var logits = tf.reshape(crnnMode, new[] { -1, 512 });
            var w = tf.Variable(tf.truncated_normal(new TensorShape(new[]{ 512, classNum }), stddev: 0.1f), name: "W");
            var b = tf.Variable(tf.constant(0f, this.classNum, "b"));

            logits = tf.matmul(logits, w) + b;

            logits = tf.reshape(logits, new[] { this.batchSize, this.classNum });
            //
            var netOutput = tf.transpose(logits, new[] { 1, 0, 2 });
            //
            var outRes = Tensorflow.Operations.gen_ops.c_t_c_greedy_decoder(netOutput, seqLength);
            Tensor[] tensors = new[] { outRes.decoded_indices, outRes.decoded_shape, outRes.decoded_values };
            return (netOutput, tensors, maxCharCount);
        }

        Tensor conv2d(Tensor inputs, int filters, string padding, bool batchNorm, string name)
        {
            control_flow_ops.
            var kernel_initializer = tf.variance_scaling_initializer();
            var bias_initializer = tf.constant_initializer(value: 0);
            //Tensor top = null;
            IActivation activation = batchNorm ? null : tf.nn.relu();
            var top = tf.layers.conv2d(inputs, filters, kernel_size: new int[] { 3, 3 }, padding: padding,
                    activation: activation, kernel_initializer: kernel_initializer,
                    bias_initializer: bias_initializer, name: name);
            if (batchNorm)
            {
                var training = tf.placeholder(tf.@bool, name: "training");
                top = tf.layers.batch_normalization(top, axis: 3, trainable: this.trainable, training:training, name: name);
                top = tf.nn.relu(top, name: name + "_relu");
            }
            return top;
        }

        Tensor cnn(Tensor inputs)
        {
            var conv1 = this.conv2d(inputs, filters: 64, padding: "valid", batchNorm: false, name: "conv1");
            var conv2 = this.conv2d(conv1, filters: 64, padding: "same", batchNorm: true, name: "conv2");
            var pool1 = tf.layers.max_pooling2d(inputs: conv2, pool_size: new[] { 2 }, strides: new[] { 2, 2 }, padding: "valid");


            var conv3 = this.conv2d(pool1, filters: 128, padding: "same", batchNorm: true, name: "conv3");
            var conv4 = this.conv2d(conv3, filters: 128, padding: "same", batchNorm: true, name: "conv4");
            var pool2 = tf.layers.max_pooling2d(inputs: conv4, pool_size: new[] { 2 }, strides: new[] { 2, 1 }, padding: "valid");


            var conv5 = this.conv2d(pool2, filters: 256, padding: "same", batchNorm: true, name: "conv5");
            var conv6 = this.conv2d(conv5, filters: 256, padding: "same", batchNorm: true, name: "conv6");
            var pool3 = tf.layers.max_pooling2d(inputs: conv6, pool_size: new[] { 2 }, strides: new[] { 2, 1 }, padding: "valid");

            var conv7 = this.conv2d(pool3, filters: 512, padding: "same", batchNorm: true, name: "conv7");
            var conv8 = this.conv2d(conv7, filters: 512, padding: "same", batchNorm: true, name: "conv8");
            var pool4 = tf.layers.max_pooling2d(inputs: conv8, pool_size: new[] { 2 }, strides: new[] { 2, 1 }, padding: "valid");

            var features = tf.squeeze(pool4, axis: new[] { 1 }, name: "features");
            return features;
        }

        Tensor _rnn(Tensor inputs, Tensor seqLength)
        {
            Tensor interOutePuts = null;
            tf_with(tf.variable_scope(name: null, default_name: "bidirectional-rnn-1"), bw_scope =>
            {
                var lstmFwCell1 = new BasicLstmCell(256);
                var lstmBwCell1 = new BasicLstmCell(256);
                var interOutput = rnn.static_bidirectional_rnn(lstmFwCell1, lstmBwCell1, new[] { inputs }, seqLength, dtype: TF_DataType.TF_FLOAT);
                interOutePuts = tf.concat(interOutput.Item1, 2);
            });

            Tensor outputs = null;
            tf_with(tf.variable_scope(name: null, default_name: "bidirectional-rnn-2"), bw_scope => {

                var lstmFwCell2 = new BasicLstmCell(256);
                var lstmBwCell2 = new BasicLstmCell(256);
                var opts = rnn.static_bidirectional_rnn(lstmFwCell2, lstmBwCell2, new[] { interOutePuts }, seqLength, dtype: TF_DataType.TF_FLOAT);
                //var interOutput2 = tf.concat(interOutput.Item1, 2);
                outputs = tf.concat(opts.Item1, 2);
            });
            return outputs;
        }
    }
}
