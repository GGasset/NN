using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Libraries
{
    public class LayerVs
    {
        public int Length => weights.Count;
        public int previousLayerLength => weights.Count > 0? weights[0].Length : 0;

        internal double[] bias;
        internal List<double[]> weights;

        public LayerVs(List<double[]> weigths, double[] bias)
        {
            this.bias = bias;
            if (weigths == null)
            {
                weigths = new List<double[]>();
            }
            else
                this.weights = weigths;
        }

        public LayerVs(int length, int prevLayerLength, double defaultBias = 1, double minWeight = -1.5, double maxWeight = 1.5)
        {
            bias = new double[length];
            weights = new List<double[]>();
            for (int i = 0; i < length; i++)
            {
                bias[i] = defaultBias;
                weights.Add(new double[prevLayerLength]);
                for (int j = 0; j < prevLayerLength; j++)
                {
                    weights[i][j] = GenerateWeight(minWeight, maxWeight);
                }
            }
        }

        internal void SubtractVs(LayerVs layerVs, double learningRate)
        {
            for (int i = 0; i < Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    weights[i][j] -= layerVs.weights[i][j] * learningRate;
                }
            }
            for (int i = 0; i < Length; i++)
            {
                bias[i] -= layerVs.bias[i];
            }
        }

        private static int randomI = 1;
        internal double GenerateWeight(double minValue, double maxValue)
        {
            Random r = new Random(DateTime.Now.Millisecond + randomI);
            randomI++;
            double output = r.NextDouble() * (maxValue - minValue) - minValue;
            if (output == 0)
                output = GenerateWeight(minValue, maxValue);
            return output;
        }

        public LayerVs(string str)
        {
            string[] biasWeightsStrs = str.Split(new string[] { "/" }, StringSplitOptions.None);
            string biasStr = biasWeightsStrs[0];

            string[] biasStrs = biasStr.Split(new string[] { "," }, StringSplitOptions.None);
            
            bias = new double[biasStrs.Length];
            for (int i = 0; i < biasStrs.Length; i++)
            {
                bias[i] = Convert.ToDouble(biasStrs[i]);
            }

            string networkWeightStr = biasWeightsStrs[1];
            string[] layerWeightStrs = networkWeightStr.Split(new string[] { "|" }, StringSplitOptions.None);

            weights = new List<double[]>();
            for (int i = 0; i < layerWeightStrs.Length; i++)
            {
                string[] neuronWeightstrs = layerWeightStrs[i].Split(new string[] { "," }, StringSplitOptions.None);
                weights.Add(new double[neuronWeightstrs.Length]);
                for (int j = 0; j < neuronWeightstrs.Length; j++)
                {
                    weights[i][j] = Convert.ToDouble(neuronWeightstrs[j]);
                }
            }
        }

        public override string ToString()
        {
            string output = "";
            foreach (var bias in bias)
            {
                output += $"{bias},";
            }
            output = output.Remove(output.Length - 1);
            output += "/";
            foreach (var neuronWeights in weights)
            {
                foreach (var weight in neuronWeights)
                {
                    output += $"{weight},";
                }
                output = output.Remove(output.Length - 1);
                output += "|";
            }
            output = output.Remove(output.Length - 1);
            return output;
        }

        public double[] this[int index] => weights[index];
    }
}
