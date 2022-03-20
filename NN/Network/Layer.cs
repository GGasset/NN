using NN.Libraries;
using System.Collections.Generic;

namespace NN
{
    public class Layer
    {
        public int Length => Vs.Length;
        public int PrevLayerLength => Vs.PrevLayerLength;
        public double bias => Vs.bias;
        internal LayerVs Vs;

        public Layer(int neuronCount, int prevLength, double bias = 1)
        {
            Vs = new LayerVs(neuronCount, prevLength, bias);
        }

        public Layer(LayerVs vals)
        {
            Vs = vals;
        }

        public double[] ExecuteLayer(double[] prevVs, Activation.ActivationFunctions activation)
        {
            double[] output = new double[Length];

            for (int i = 0; i < Length; i++)
                output[i] = Neuron.Execute(prevVs, Vs[i], Vs.bias, activation);

            return output;
        }

        public void GetGrads(double[] prevVs, double[] costs, Activation.ActivationFunctions activation, out double[] prevLayerCosts, out LayerVs grads)
        {
            prevLayerCosts = new double[PrevLayerLength];
            grads = new LayerVs(new List<double[]>());
            for (int neuronI = 0; neuronI < Length; neuronI++)
            {
                Neuron.GetGradients(costs[neuronI], prevVs, Vs[neuronI], bias, activation, out double[] weigthGrads, out double[] prevCosts, out double biasGrads);
                grads.AddNeuron(weigthGrads, biasGrads);
                for (int prevLayerI = 0; prevLayerI < PrevLayerLength; prevLayerI++)
                {
                    prevLayerCosts[prevLayerI] += prevCosts[prevLayerI];
                }
            }
        }

        public override string ToString()
        {
            return Vs.ToString();
        }

        public Layer(string str)
        {
            this.Vs = new LayerVs(str);
        }

        public static double[] GetRandomWeigths(int count)
        {
            double[] output = new double[count];
            for (int i = 0; i < count; i++)
                output[i] = Neuron.GetRandomWeigth();
            return output;
        }
    }
}
