using System;
using NN.Libraries;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN
{
    public static class Neuron
    {
        public static double Execute(double[] prevVs, double[] weigths, double bias, Activation.ActivationFunctions activation)
        {
            if (prevVs.Length != weigths.Length)
                throw new IndexOutOfRangeException();

            double output = bias;
            for (int i = 0; i < prevVs.Length; i++)
                output += prevVs[i] * weigths[i];
            output = Activation.Activate(output, activation);

            return output;
        }


        public static void GetGradients(double cost, double[] prevVs, double[] weigths, double bias, Activation.ActivationFunctions activation, out double[] weigthsGrads, out double[] prevActGrads, out double biasGrad)
        {
            if (prevVs.Length != weigths.Length)
                throw new IndexOutOfRangeException();

            double output = bias;
            for (int i = 0; i < prevVs.Length; i++)
                output += prevVs[i] * weigths[i];


            cost *= Derivatives.DerivativeOf(output, activation);
            biasGrad = cost;

            weigthsGrads = new double[weigths.Length];
            prevActGrads = new double[prevVs.Length];
            for (int i = 0; i < prevVs.Length; i++)
            {
                weigthsGrads[i] = prevVs[i] * cost;
                prevActGrads[i] = weigths[i] * cost;
            }
        }

        internal static int rI = 0;
        public static double GetRandomWeigth()
        {
            rI++;
            Random r = new Random(rI);
            bool isPositive = Convert.ToBoolean(Math.Round(r.NextDouble()));
            return 1 + -2 * Convert.ToInt32(isPositive) + r.NextDouble() * r.Next(0, 3);
        }
    }
}
