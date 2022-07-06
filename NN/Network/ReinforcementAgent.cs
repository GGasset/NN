using System;
using NN;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NN.Libraries;
using static NN.Libraries.Activation;

namespace NN
{
    public class ReinforcementAgent
    {
        public NN n;
        List<double[]> inputs;
        List<List<double[]>> neuronActivations;
        List<List<double[]>> neuronLinears;
        internal List<double> rewards;
        internal double reward;
        internal double learningRate;

        public ReinforcementAgent(int inputLength, int[] topology, double learningRate, ActivationFunctions activation)
        {
            n = new NN(inputLength, topology, activation);
            reward = 0;
            rewards = new List<double>();
            inputs = new List<double[]>();
            neuronActivations = new List<List<double[]>>();
            this.learningRate = learningRate;
        }

        public ReinforcementAgent(string str, double learningRate)
        {
            this.learningRate = learningRate;
            reward = 0;
            rewards = new List<double>();
            inputs = new List<double[]>();
            neuronActivations = new List<List<double[]>>();
            n = new NN(str);
        }

        internal double[] ExecuteAgent(double[] input)
        {
            rewards.Add(reward);
            inputs.Add(input);
            double[] output = n.ExecuteNetwork(input, out List<double[]> linears, out List<double[]> neuronOutputs);
            neuronLinears.Add(linears);
            neuronActivations.Add(neuronOutputs);
            return output;
        }

        internal void TerminateAgent()
        {
            List<List<LayerVs>> grads = new List<List<LayerVs>>();
            for (int t = 0; t < inputs.Count; t++)
            {
                int outputLength = n.layers[n.layers.Count - 1].Length;
                double[] costs = new double[outputLength];
                for (int outputI = 0; outputI < outputLength; outputI++)
                {
                    double[] output = neuronActivations[t][neuronActivations[t].Count - 1];
                    costs[outputI] = Derivatives.LogLikelyhoodTermDerivative(output[outputI], rewards[t]);
                }
                grads.Add(n.GetGradients(costs, inputs[t], neuronLinears[t], neuronActivations[t]));
            }

            foreach (var grad in grads)
                n.SubtractGrads(grad, learningRate);

            reward = 0;
            rewards = new List<double>();
            inputs = new List<double[]>();
            neuronActivations = new List<List<double[]>>();
            neuronLinears = new List<List<double[]>>();
        }

        internal void GiveReward(double reward)
        {
            this.reward += reward;
        }

        internal void AddToLastReward(double reward)
        {
            rewards[rewards.Count - 1] += reward;
        }

        internal void ChangeLastReward(double reward)
        {
            rewards[rewards.Count - 1] = reward;
        }
    }
}
