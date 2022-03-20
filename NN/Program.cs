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
            NN n = new NN(2, new int[] { 5, 3, 1 }, Libraries.Activation.ActivationFunctions.Sine);
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

            Random r = new Random(65456453);

            for (int i = 0; i < 20000; i++)
            {
                int trainingI = r.Next(0, X.Count);
                List<LayerVs> Vs = n.GetSupervisedGrads(X[trainingI], Y[trainingI], NN.CostFunctions.SquaredMean);
                n.SubtractGrads(Vs, .1);
            }

            for (int x = 0; x < 10; x++)
            {
                for (int y = 0; y < 10; y++)
                {
                    double currentX = x / 10.0;
                    double currentY = y / 10.0;

                    Console.Write($"{n.ExecuteNetwork(new double[] { currentX, currentY })[0]}  ");
                }
                Console.WriteLine();
            }

            Console.ReadKey();
        }
    }
}
