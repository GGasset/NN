using NN.Libraries;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN
{
    public class Layer
    {
        public int Length => vals.Length;
        internal LayerVs vals;

        public double[] ExecuteLayer(double[] prevVs, Activation.ActivationFunctions activation)
        {
            double[] output = new double[Length];

            for (int i = 0; i < Length; i++)
                output[i] = Neuron.Execute(prevVs, vals[i], vals.bias, activation);

            return output;
        }


    }
}
