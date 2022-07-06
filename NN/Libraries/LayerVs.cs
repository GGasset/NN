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

        public LayerVs(List<double[]> weigths, double bias = 0)
        {
            this.bias = bias;
            if (weigths == null)
            {
                weigths = new List<double[]>();
            }
            else
                this.weigths = weigths;
        }

        internal void SubtractVs(LayerVs layerVs)
        {
            if (layerVs.weigths.Count != weigths.Count)
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

                for (int l = 0; l < weigths[i].Length; l++)
                {
                    weigths[i][l] -= layerVs.weigths[i][l];
                }
            }

            this.bias -= layerVs.bias;
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
