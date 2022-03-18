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

        internal double bias;
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

        internal void AddVs(LayerVs layerVs)
        {
            if (layerVs.weigths.Count != weigths.Count)
            {
                throw new IndexOutOfRangeException();
            }

            for (int i = 0; i < weigths.Count; i++)
            {
                if (weigths[i].Length != layerVs.weigths[i].Length)
                {
                    throw new IndexOutOfRangeException();
                }

                for (int l = 0; l < weigths[i].Length; l++)
                {
                    weigths[i][l] += layerVs.weigths[i][l];
                }
            }

            this.bias += layerVs.bias;
        }
    }
}
