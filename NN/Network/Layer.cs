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

        public Layer(LayerVs values)
        {
            vals = values;
        }

        public Layer(int length, int previousLayerLength, double defaultBias, double minWeight, double maxWeight)
        {
            vals = new LayerVs(length, previousLayerLength, defaultBias, minWeight, maxWeight);
        }

        public double[] ExecuteLayer(double[] prevVs, Activation.ActivationFunctions activation, out double[] linearFunctions)
        {
            double[] output = new double[Length];
            linearFunctions = new double[Length];
            for (int i = 0; i < Length; i++)
            {
                output[i] = Neuron.Execute(prevVs, vals[i], vals.bias[i], activation, out double currentLinear);
                linearFunctions[i] = currentLinear;
            }

            return output;
        }

        public void GetGradients(double[] costs, double[] prevVs, double[] linearFunctions, Activation.ActivationFunctions activationFunction, out List<double[]> weightGrads, out double[] prevActivationGrads, out double[] biasGrads)
        {
            weightGrads = new List<double[]>();
            biasGrads = new double[Length];
            prevActivationGrads = new double[prevVs.Length];
            
            for (int i = 0; i < Length; i++)
            {
                Neuron.GetGradients(costs[i], prevVs, vals.weigths[i], vals.bias[i], activationFunction, linearFunctions[i], out double[] neuronWeightGrads, out double[] currentPrevActivationsGrads, out double biasGrad);
                weightGrads.Add(neuronWeightGrads);
                biasGrads[i] = biasGrad;

                for (int j = 0; j < prevVs.Length; j++)
                {
                    prevActivationGrads[j] += currentPrevActivationsGrads[j];
                }
            }
        }
    }
}
