using System;
using NN.Libraries;

namespace NN
{
    public static class Neuron
    {
        public static double Execute(double[] prevVs, double[] weigths, double bias, Activation.ActivationFunctions activation, out double linearFunction)
        {
            if (prevVs.Length != weigths.Length)
                throw new IndexOutOfRangeException();

            double output = 0;
            for (int i = 0; i < prevVs.Length; i++)
                output += prevVs[i] * weigths[i];
            linearFunction = output;
            output += bias;
            return Activation.Activate(output, activation);
        }


        public static void GetGradients(double cost, double[] prevVs, double[] weigths, double bias, Activation.ActivationFunctions activation, double linearFunction, out double[] weigthsGrads, out double[] prevActGrads, out double biasGrad)
        {
            cost *= Derivatives.DerivativeOf(linearFunction, activation);
            biasGrad = cost;

            weigthsGrads = new double[weigths.Length];
            prevActGrads = new double[prevVs.Length];
            for (int i = 0; i < prevVs.Length; i++)
            {
                weigthsGrads[i] = prevActGrads[i] * cost;
                prevActGrads[i] = weigths[i] * cost;
            }
        }
    }
}
