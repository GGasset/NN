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
        internal List<Layer> layers;

        /// <param name="topology">neuronCount at each layer, this includes output</param>
        public NN(int inputLength, int[] topology)
        {
            layers = new List<Layer>();
            int currentPrevLength = inputLength;
            for (int i = 0; i < topology.Length; i++)
            {
                layers.Add(new Layer(topology[i], currentPrevLength));
                currentPrevLength = topology[i];
            }
        }

        public NN(List<Layer> layers)
        {
            this.layers = layers;
        }

        public enum CostFunctions
        {
            SquaredMean,
            BinaryCrossEntropy,
            logLikelyhoodTerm,
        }
    }
}
