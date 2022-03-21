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
        NN n;
        List<double[]> inputs;
        List<List<double[]>> neuronActivations;
        List<double> rewards;
        double reward;
        double learningRate;

        public ReinforcementAgent(int inputLength, int[] topology, double learningRate, ActivationFunctions activation)
        {
            n = new NN(inputLength, topology, activation);
            reward = 0;
            rewards = new List<double>();
            inputs = new List<double[]>();
            neuronActivations = new List<List<double[]>>();
            this.learningRate = learningRate;
        }

        public double[] ExecuteAgent(double[] input)
        {
            rewards.Add(reward);
            inputs.Add(input);
            neuronActivations.Add(n.ExecuteNetwork(input, out double[] output));
            return output;
        }

        public void TerminateAgent()
        {
            List<List<LayerVs>> grads = new List<List<LayerVs>>();
            for (int t = 0; t < inputs.Count; t++)
            {
                int outputLength = neuronActivations[t][neuronActivations[t].Count - 1].Length;
                double[] costs = new double[outputLength];
                for (int outputI = 0; outputI < outputLength; outputI++)
                {
                    double[] output = neuronActivations[t][neuronActivations[t].Count - 1];
                    costs[outputI] = Derivatives.LogLikelyhoodTermDerivative(output[outputI], rewards[t]);
                }
               grads.Add(n.GetGrads(inputs[t], neuronActivations[t], costs));
            }

            foreach (var grad in grads)
                n.SubtractGrads(grad, learningRate);

            reward = 0;
            rewards = new List<double>();
            inputs = new List<double[]>();
            neuronActivations = new List<List<double[]>>();
        }

        public void GiveReward(double reward)
        {
            this.reward += reward;
        }
    }
}
