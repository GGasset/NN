using NN.Libraries;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN
{
    class Program
    {
        static void Main(string[] args)
        {
            NN n = new NN(2, new int[] { 3, 4, 4, 1 }, Activation.ActivationFunctions.Sine);
            List<double[]> X = new List<double[]>();
            List<double[]> Y = new List<double[]>();
            X.Add(new double[] { 0, 0 });
            Y.Add(new double[] { 1 });

            X.Add(new double[] { 1, 1 });
            Y.Add(new double[] { 1 });

            X.Add(new double[] { 1, 0 });
            Y.Add(new double[] { -1 });

            X.Add(new double[] { 0, 1 });
            Y.Add(new double[] { -1 });

            X.Add(new double[] { .5, .5 });
            Y.Add(new double[] { 0 });


            for (int i = 0; i < 10000; i++)
            {
                n.Supervisedbatch(X, Y, 5, .01, NN.CostFunctions.SquaredMean, out double averageCost);
                //Console.WriteLine("Average Cost = " + averageCost);
            }

            n = new NN(n.ToString());

            for (int x = 0; x < 5; x++)
            {
                for (int y = 0; y < 5; y++)
                {
                    double currentX = x / 5.0;
                    double currentY = y / 5.0;

                    Console.Write($"{n.ExecuteNetwork(new double[] { currentX, currentY })[0]}  ");
                }
                Console.WriteLine();
            }

            Console.ReadKey();
        }
    }
}
