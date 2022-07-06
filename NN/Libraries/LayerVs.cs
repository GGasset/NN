using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Libraries
{
    public class LayerVs
    {
        public int Length => weigths.Count;
        public int PrevLayerLength => weigths.Count > 0? weigths[0].Length : 0;

        internal double[] bias;
        internal List<double[]> weigths;

        public LayerVs(List<double[]> weigths, double[] bias)
        {
            this.bias = bias;
            this.weigths = weigths;
        }

        public LayerVs(int length, int previousLayerLength, double defaultBias, double minWeight, double maxWeight)
        {
            bias = new double[length];
            weigths = new List<double[]>();
            for (int i = 0; i < length; i++)
            {
                bias[i] = defaultBias;
                weigths.Add(new double[previousLayerLength]);
                for (int j = 0; j < previousLayerLength; j++)
                {
                    weigths[i][j] = GenerateWeight(minWeight, maxWeight);
                }
            }
        }

        internal void SubtractVs(LayerVs layerVs, double learningRate)
        {
            for (int i = 0; i < Length; i++)
            {
                for (int j = 0; j < weigths[i].Length; j++)
                {
                    weigths[i][j] -= layerVs.weigths[i][j] * learningRate;
                }
            }

            for (int i = 0; i < Length; i++)
            {
                this.bias[i] -= layerVs.bias[i] * learningRate;
            }
        }

        internal void SubtractVs(List<double[]> weigths, double[] bias, double learningRate)
        {
            LayerVs toSubtract = new LayerVs(weigths, bias);
            SubtractVs(toSubtract, learningRate);
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

        public double[] this[int index] => weigths[index];
    }
}
