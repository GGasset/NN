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
        Activation.ActivationFunctions activationFunction;

        /// <param name="topology">neuronCount at each layer, this includes output</param>
        public NN(int inputLength, int[] topology, Activation.ActivationFunctions activationFunc)
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
            activationFunction = activationFunc;
        }

        public double[] ExecuteNetwork(double[] input)
        {
            ExecuteNetwork(input, out double[] output);
            return output;
        }

        /// <param name="output">last layer of execution results</param>
        /// <returns>All executions results</returns>
        public List<double[]> ExecuteNetwork(double[] input, out double[] output)
        {
            List<double[]> outputs = new List<double[]>();
            double[] prevLayerOuts = input;

            for (int layerI = 0; layerI < layers.Count; layerI++)
            {
                outputs.Add(prevLayerOuts = layers[layerI].ExecuteLayer(prevLayerOuts, activationFunction));
            }

            output = outputs[layers.Count - 1];

            if (outputs.Count != layers.Count)
                throw new IndexOutOfRangeException();
            return outputs;
        }

        public List<LayerVs> GetGrads(double[] input, double[] costs)
        {
            List<LayerVs> grads = new List<LayerVs>();
            List<double[]> layerOuts = ExecuteNetwork(input, out _);
            for (int layerI = Length - 1; layerI >= 0; layerI--)
            {
                layers[layerI].GetGrads(layerI > 0 ? layerOuts[layerI - 1] : input, costs, activationFunction, out costs, out LayerVs layerGrads);
                grads.Add(layerGrads);
            }
            return grads;
        }

        public enum CostFunctions
        {
            SquaredMean,
            BinaryCrossEntropy,
            logLikelyhoodTerm,
        }
    }
}
