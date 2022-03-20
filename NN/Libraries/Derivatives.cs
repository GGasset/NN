using System;
using static NN.NN;
using static NN.Libraries.Activation;

namespace NN.Libraries
{
    public static class Derivatives
    {
        public static double DerivativeOf(double neuronLinear, ActivationFunctions activation)
        {
            switch (activation)
            {
                case ActivationFunctions.Relu:
                    return ReluDerivative(neuronLinear);
                case ActivationFunctions.Sigmoid:
                    return SigmoidDerivative(neuronLinear);
                case ActivationFunctions.Tanh:
                    return TanhDerivative(neuronLinear);
                case ActivationFunctions.Sine:
                    return SinDerivative(neuronLinear);
                default:
                    throw new NotImplementedException();
            }
        }

        /// <param name="expected">In case of Reinforcement learning/logLikelyhood expected is reward</param>
        public static double DerivativeOf(double neuronActivation, double expected, CostFunctions costFunction)
        {
            if (double.IsNaN(expected))
            {
                return 0;
            }
            switch (costFunction)
            {
                case CostFunctions.BinaryCrossEntropy:
                    throw new NotImplementedException();
                case CostFunctions.SquaredMean:
                    return SquaredMeanErrorDerivative(neuronActivation, expected);
                case CostFunctions.logLikelyhoodTerm:
                    return LogLikelyhoodTermDerivative(neuronActivation, expected);
                default:
                    throw new NotImplementedException();
            }
        }

        public static double LogLikelyhoodTermDerivative(double output, double reward) => -(1 / output * Math.Log(10)) * -Math.Log10(output) + reward;

        public static double SquaredMeanErrorDerivative(double neuronOutput, double expectedOutput) => 2 * (neuronOutput - expectedOutput);

        //public static double BinaryCrossEntropyDerivative(double neuronOutput, double expectedOutput) =>  /

        public static double SigmoidDerivative(double neuronActivation) => Activation.SigmoidActivation(neuronActivation) * (1 - SigmoidDerivative(neuronActivation));

        /// <param name="connectedNeuronActivation">Activation Connected to the weigth that is being computed</param>
        public static double LinearFunctionDerivative(double connectedNeuronActivation) => connectedNeuronActivation;

        public static int ReluDerivative(double neuronActivation) => neuronActivation > 0 ? 1 : 0;

        public static double TanhDerivative(double neuronActivation) => 1 - Math.Pow(Activation.TanhActivation(neuronActivation), 2);

        public static double SinDerivative(double neuronActivation) => Math.Cos(neuronActivation);

        public static double MultiplicationDerivative(double a, double aDerivative, double b, double bDerivative) => a * aDerivative + b * bDerivative;

        public static double SumDerivative(double aDerivative, double bDerivative) => aDerivative + bDerivative;

    }
}
