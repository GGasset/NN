﻿using System;
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

        public LayerVs(int neuronCount, int prevLength, double bias = 1)
        {
            this.bias = bias;
            weigths = new List<double[]>();
            for (int i = 0; i < neuronCount; i++)
                weigths.Add(Layer.GetRandomWeigths(prevLength));
        }

        public LayerVs(List<double[]> weigths, double bias = 0)
        {
            this.bias = bias;
            if (weigths == null)
                this.weigths = new List<double[]>();
            else
                this.weigths = weigths;
        }

        internal void AddNeuron(double[] weigths, double biasAddition = 0)
        {
            if (weigths.Length != PrevLayerLength)
            {
                throw new IndexOutOfRangeException();
            }
            this.weigths.Add(weigths);
            bias += biasAddition;
        }

        internal void SubtractVs(LayerVs layerVs)
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
                    weigths[i][l] -= layerVs.weigths[i][l];
                }
            }

            this.bias -= layerVs.bias;
        }

        internal void SubtractVs(List<double[]> weigths, double bias)
        {
            LayerVs toSubtract = new LayerVs(weigths, bias);
            SubtractVs(toSubtract);
        }

        public double[] this[int index] => weigths[index];
    }
}
