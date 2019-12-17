using NumSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using static Tensorflow.Binding;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using NumSharp.Backends.Unmanaged;
using System.Runtime.CompilerServices;
using NumSharp.Backends;

namespace TfConsoleApp
{
    public class HandleWriteTest
    {
        private Session session;
        private Saver saver;
        private int inputImgHeight;
        private int inputImgWidth;
        private int testBatchSize;
        private Tensor inputTensor;
        private Tensor seqLengthTensor;
        private int maxCharCount = 0;
        private Tensor denseDecode;
        private List<string> wordsList;
        private List<int> wordsOnehotList;
        private string modelSavePath;
        private int classNum;
        

        public HandleWriteTest(string modelPath, string wordOnehotFilePath, int classNums,  int testBatchSize,
            int inputImgHeight, int inputImgWidth)
        {
            this.inputImgHeight = inputImgHeight;
            this.inputImgWidth = inputImgWidth;
            this.modelSavePath = modelPath;
            this.testBatchSize = testBatchSize;
            this.classNum = classNums;
            this.ReadWordDictFromFile(wordOnehotFilePath);
            this.inputTensor = tf.placeholder(TF_DataType.TF_FLOAT, new TensorShape(new[] { this.testBatchSize, this.inputImgHeight, this.inputImgWidth, 1}));
            this.seqLengthTensor = tf.placeholder(TF_DataType.TF_INT32, null, name: "seq_len");

            var crnnNet = new CRNN(this.inputTensor, this.seqLengthTensor, this.testBatchSize, this.inputImgHeight, this.inputImgWidth, classNum, true);
            var (netOut, decoded, maxCharCount) = crnnNet.ConstructGraph();
            this.denseDecode = Tensorflow.Operations.gen_ops.sparse_to_dense(decoded[0], decoded[1], decoded[2], default_value: new Tensor(-1));

            session = tf.Session();
            saver = tf.train.Saver();
            saver.restore(session, modelPath);
        }

        private void ReadWordDictFromFile(string path)
        {
            if (File.Exists(path))
            {
                this.wordsList = new List<string>();
                this.wordsOnehotList = new List<int>();
                string text = File.ReadAllText(path);
                if (!string.IsNullOrWhiteSpace(text))
                {
                    string[] tv = text.Split(new char[] { '{', '}', ',' }, StringSplitOptions.RemoveEmptyEntries);
                    foreach (var i in tv)
                    {
                        string[] pair = i.Split(new char[] { '\'', ':', ' ' }, StringSplitOptions.RemoveEmptyEntries);
                        if(pair.Length == 2)
                        {
                            try
                            {
                                this.wordsOnehotList.add(Convert.ToInt32(pair[1]));
                                this.wordsList.add(pair[0]);
                            }
                            catch { }
                        }
                    }
                }
            }
        }

        public string[] TestImg(string[] imgPathList)
        {
            var (batchData, batchSize, imgList) = this.getInputImgs(imgPathList);
            if (batchSize != this.testBatchSize)
                throw new ArgumentException($"网络构建 batch size {this.testBatchSize} 和实际输入的 batch size {batchSize} 不一样");
            //this.session.run()
            var maxCharData = new NDArray(NPTypeCode.Int32, batchSize) * this.maxCharCount;
            FeedItem[] feedDict = new FeedItem[] { new FeedItem(this.inputTensor, batchData),
                new FeedItem(this.seqLengthTensor,  maxCharData)};
            var predict = this.session.run(denseDecode, feedDict);
            var predict_seq = this.PredictToWords(predict);
            return predict_seq;
        }

        private string[] PredictToWords(NDArray decoded)
        {
            List<String> words = new List<string>();
            foreach (var seq in decoded)
            {
                var seq_words = "";
                foreach (var onehot in (seq as NDArray))
                {
                    if ((int)onehot == 0)
                    {
                        break;
                    }
                    seq_words += wordsList[wordsOnehotList.IndexOf((int)onehot)];
                }
                words.append(seq_words);
            }
            return words.ToArray();
        }

        public (NDArray, int, List<Mat>) getInputImgs(string[] imgPathList)
        {
            var batchSize = imgPathList.Length;
            var batchData = np.zeros(new[] { batchSize, inputImgHeight, inputImgWidth, 1 });
            var imgList = new List<Mat>();
            for (int i = 0; i < batchSize; i++)
            {
                var img = Cv2.ImRead(imgPathList[i], ImreadModes.Grayscale);
                imgList.add(img);
                var resizedImg = this.resizeImg(img);
                var reshapImg = resizedImg.Reshape(1, new[] { this.inputImgHeight, this.inputImgWidth, 1 });
                var imgNorm = reshapImg / 255 * 2 - 1;
                batchData[i] = WrapWithNDArray(imgNorm.ToMat());//ConvertMatToArray(imgNorm.ToMat());
            }
            return (batchData, batchSize, imgList);
        }

        private int[,,] ConvertMatToArray(Mat img)
        {
            int[,,] data = new int[img.Height,img.Width,1];
            for (int i = 0; i < img.Height; i++)
                for (int j = 0; j < img.Width; j++)
                    data[i,j,0] = img.At<int>(i, j);
            return data;

        }

        private Mat resizeImg(Mat img)
        {
            Mat outImg = new Mat();
            var size = img.Size();
            if (size.Width > this.inputImgWidth)
            {
                size.Width = this.inputImgWidth;
                var ratio = (float)this.inputImgWidth / size.Width;
                Cv2.Resize(img, outImg, size);
            }
            else
            {
                outImg = new Mat(size:new Size(this.inputImgWidth, this.inputImgHeight), type:img.Type());
                var ratio = this.inputImgHeight / size.Height;
                Cv2.Resize(img, outImg, new Size(size.Width * ratio, this.inputImgHeight));
            }
            return outImg;
        }

        //this method copies Mat to a new NDArray
        public static unsafe Tensor ToTensor(Mat src)
        {
            Shape shape = (1, src.Height, src.Width, src.Type().Channels);
            IntPtr handle;
            var tensor = new Tensor(handle = c_api.TF_AllocateTensor(TF_DataType.TF_UINT8, shape.Dimensions.Select(d => (long)d).ToArray(), shape.NDim, (ulong)shape.Size));

            new UnmanagedMemoryBlock<byte>(src.DataPointer, shape.Size)
                .CopyTo((byte*)c_api.TF_TensorData(handle));

            return tensor;
        }

        //this method wraps without copying Mat. Although in some cases (randomally) Tensorflow might decide it has to copy.
        public static unsafe Tensor WrapToTensor(Mat src)
        {
            Shape shape = (1, src.Height, src.Width, src.Type().Channels);
            var storage = new UnmanagedStorage(new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(src.DataPointer, shape.Size, () => Donothing(src))), shape); //we pass donothing as it keeps reference to src preventing its disposal by GC
            return new Tensor(new NDArray(storage));
        }

        [MethodImpl(MethodImplOptions.NoOptimization)]
        private static void Donothing(Mat m)
        {
            var a = m;
        }

        //this method copies Mat to a new NDArray
        public static unsafe NDArray ToNDArray(Mat src)
        {
            var nd = new NDArray(NPTypeCode.Byte, (1, src.Height, src.Width, src.Type().Channels), fillZeros: false);
            new UnmanagedMemoryBlock<byte>(src.DataPointer, nd.size)
                .CopyTo(nd.Unsafe.Address);

            return nd;
        }

        //this method wraps without copying Mat.
        public static unsafe NDArray WrapWithNDArray(Mat src)
        {
            Shape shape = (1, src.Height, src.Width, src.Type().Channels);
            var storage = new UnmanagedStorage(new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(src.DataPointer, shape.Size, () => Donothing(src))), shape); //we pass donothing as it keeps reference to src preventing its disposal by GC
            return new NDArray(storage);
        }
    }
}
