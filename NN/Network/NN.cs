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
        public readonly Activation.ActivationFunctions activationFunction;

        /// <param name="topology">neuronCount at each layer, this includes output layer</param>
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
            if (input.Length != inputLength)
                throw new IndexOutOfRangeException();

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

        public void Supervisedbatch(List<double[]> X, List<double[]> Y, int batchLength, double learningRate, CostFunctions costFunction, out double averageCost)
        {
            if (batchLength < 1)
            {
                averageCost = 0;
                return;
            }

            if (X.Count != Y.Count)
                throw new IndexOutOfRangeException();

            List<List<LayerVs>> grads = new List<List<LayerVs>>();

            Random r = new Random(Neuron.rI + DateTime.Now.Millisecond);
            Neuron.rI++;
            averageCost = 0;
            for (int i = 0; i < batchLength; i++)
            {
                int trainingI = r.Next(X.Count);

                grads.Add(GetSupervisedGrads(X[trainingI], Y[trainingI], costFunction, out double cost));
                averageCost += cost;
            }
            averageCost /= batchLength;


            foreach (var grad in grads)
                SubtractGrads(grad, learningRate);
        }

        public List<LayerVs> GetSupervisedGrads(double[] input, double[] expected, CostFunctions costFunction, out double cost)
        {
            List<double[]> neuronActivations = ExecuteNetwork(input, out double[] output);
            if (expected.Length != output.Length)
                throw new IndexOutOfRangeException("Expected output count is incorrect");

            double[] costs = new double[output.Length];
            for (int i = 0; i < output.Length; i++)
                costs[i] = Derivatives.DerivativeOf(output[i], expected[i], costFunction);
            cost = Cost.GetCostOf(output, expected, costFunction);

            return GetGrads(input, neuronActivations, costs);
        }

        public List<LayerVs> GetGrads(double[] input, List<double[]> layerOuts, double[] costs)
        {
            List<LayerVs> grads = new List<LayerVs>();
            for (int layerI = Length - 1; layerI >= 0; layerI--)
            {
                layers[layerI].GetGrads(layerI > 0 ? layerOuts[layerI - 1] : input, costs, activationFunction, out costs, out LayerVs layerGrads);
                grads.Insert(0, layerGrads);
            }
            return grads;
        }

        public void SubtractGrads(List<LayerVs> Vs, double learningRate)
        {
            if (Vs.Count != Length)
                throw new IndexOutOfRangeException();

            for (int i = 0; i < Length; i++)
                layers[i].Vs.SubtractVs(Vs[i], learningRate);
        }

        public override string ToString()
        {
            string str = "network:\n";
            foreach (var layer in layers)
                str += layer;
            return str;
        }

        public NN(string str)
        {
            str = str.Replace("network:\n", "");
            string[] layers = str.Split(new string[] {"layer:"}, StringSplitOptions.RemoveEmptyEntries);
            this.layers = new List<Layer>();
            foreach (var layer in layers)
                this.layers.Add(new Layer(layer));
        }

        public enum CostFunctions
        {
            SquaredMean,
            BinaryCrossEntropy,
            logLikelyhoodTerm,
        }
    }
}
