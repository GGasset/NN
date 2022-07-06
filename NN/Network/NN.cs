using System;
using NN.Libraries;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN
{
    public class NN
    {
        public int Length => layers.Count;
        public int inputLength => layers.Count > 0 ? layers[0].PrevLayerLength : 0;
        internal List<Layer> layers;
        public int Length => layers.Count;
        public int InputLength => layers[0].vals.weigths.Count;
        private readonly Activation.ActivationFunctions ActivationFunction;

        public NN(List<Layer> layers, Activation.ActivationFunctions activationFunction)
        {
            layers = new List<Layer>();
            int currentPrevLength = inputLength;
            activationFunction = activationFunc;
            for (int i = 0; i < topology.Length; i++)
            {
                layers.Add(new Layer(topology[i], currentPrevLength));
                currentPrevLength = topology[i];
            }
        }
        public NN(List<Layer> layers, Activation.ActivationFunctions activationFunc)
        {
            this.layers = layers;
            ActivationFunction = activationFunction;
        }

        public NN(int inputLength, int[] lengths, Activation.ActivationFunctions activationFunction, double defaultBias = 1, double minWeight = -1.5, double maxWeight = 1.5)
        {
            ActivationFunction = activationFunction;
            layers = new List<Layer>();
            for (int i = 0; i < lengths.Length; i++)
            {
                int previousLayerLength = i == 0 ? inputLength : lengths[i - 1];
                layers.Add(new Layer(lengths[i], previousLayerLength, defaultBias, minWeight, maxWeight));
            }
        }

        public double[] ExecuteNetwork(double[] input, out List<double[]> linearFunctions, out List<double[]> neuronOutputs)
        {
            if (InputLength != input.Length)
                throw new Exception("Input length is different from network input length");

            linearFunctions = new List<double[]>();
            neuronOutputs = new List<double[]>();
            for (int i = 0; i < Length; i++)
            {
                input = layers[i].ExecuteLayer(input, ActivationFunction, out double[] layerLinearFunctions);
                linearFunctions.Add(layerLinearFunctions);
                neuronOutputs.Add(input);
            }
            return input;
        }

        public List<LayerVs> GetSupervisedGradients(double[] X, double[] y, Cost.CostFunctions costFunction, out double cost)
        {
            double[] output = ExecuteNetwork(X, out List<double[]> linearFunctions, out List<double[]> neuronOutputs);
            if (output.Length != y.Length)
                throw new ArgumentOutOfRangeException();

            cost = Cost.GetCost(output, y, costFunction);
            double[] costGrads = Derivatives.DerivativeOf(output, y, costFunction);

            return GetGradients(costGrads, X, linearFunctions, neuronOutputs);
        }

        public List<LayerVs> GetGradients(double[] costGrads, double[] input, List<double[]> linearFunctions, List<double[]> neuronOutputs)
        {
            List<LayerVs> outputGradients = new List<LayerVs>();
            for (int i = Length - 1; i >= 0; i--)
            {
                double[] prevLayerActivations = i == 0 ? input : neuronOutputs[i - 1];
                layers[i].GetGradients(costGrads, prevLayerActivations, linearFunctions[i], ActivationFunction, out List<double[]> weightGrads, out double[] prevActivationGrads, out double[] biasGrads);
                
                costGrads = prevActivationGrads;
                outputGradients.Add(new LayerVs(weightGrads, biasGrads));
            }
            return outputGradients;
        }

        internal void SubtractGrads(List<LayerVs> grads, double learningRate)
        {
            for (int i = 0; i < Length; i++)
                layers[i].vals.SubtractVs(grads[i], learningRate);
        }
    }
}
