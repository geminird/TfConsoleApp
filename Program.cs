using System;
using System.Collections.Generic;

namespace TfConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            HandleWriteTest test = new HandleWriteTest("./model/ckpt",
                "./data/word_onehot.txt",
                2675, 28, 32, 900);
            List<string> imgList = new List<string>();
            List<string> nameList = new List<string>();
            for (int i = 0; i < 28; i++)
            {
                imgList.Add(string.Format("./samples/{0:5D}.jpg", (i + 32)));
                nameList.Add(string.Format("{0:5D}", (i + 32)));
                if ((i + 1) % 28 == 0)
                {
                    var res = test.TestImg(imgList.ToArray());
                }
            }

            Console.WriteLine("Hello World!");
        }
    }
}
