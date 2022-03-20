using System;
using System.Collections.Generic;
using static NN.NN;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Libraries
{
    public static class Cost
    {
        public static List<double> GetCostGradients(double[] output, double[] expected, CostFunctions costFunction, out double cost)
        {
            cost = GetCostOf(output, expected, costFunction);

            List<double> Gradients = new List<double>();
            for (int i = 0; i < Math.Min(output.Length, expected.Length); i++)
            {
                Gradients.Add(Derivatives.DerivativeOf(output[i], expected[i], costFunction));
            }
            return Gradients;
        }

        public static double GetCostOf(double[] output, double[] expected, CostFunctions costFunction)
        {
            switch (costFunction)
            {
                case CostFunctions.SquaredMean:
                    return SquaredMeanEmpiricalLoss(output, expected);
                case CostFunctions.BinaryCrossEntropy:
                    return BinaryCrossEntropyEmpiricalLoss(output, expected);
                default:
                    throw new NotImplementedException();
            }
        }

        public static double SquaredMeanEmpiricalLoss(double[] output, double[] expected)
        {
            if (output.Length != expected.Length)
                throw new Exception();
            double sum = 0;
            for (int i = 0; i < output.Length; i++)
            {
                if (!double.IsNaN(expected[i]))
                {
                    double currentCost = expected[i] - output[i];
                    sum += currentCost * currentCost;
                }
            }
            sum /= output.Length;

            return sum;
        }

        /// <summary>
        /// Used to train the network for boolean outputs
        /// </summary>
        public static double BinaryCrossEntropyEmpiricalLoss(double[] output, double[] expected)
        {
            if (output.Length != expected.Length)
                throw new Exception();
            double sum = 0;
            for (int i = 0; i < output.Length; i++)
                if (!double.IsNaN(expected[i]))
                    sum += 1 - expected[i] * Math.Log(1 - output[i]) + expected[i] * Math.Log(expected[i]);
            sum /= output.Length;

            return sum;
        }
    }
}
