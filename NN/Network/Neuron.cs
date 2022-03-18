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

            return Activation.Activate(output, activation);
        }
    }
}
