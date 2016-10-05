package net.benwoodworth.neuralnetworks.network

/**
 * A neuron in a neural network.
 */
interface Neuron {
    /**
     * Evaluate the neuron.
     */
    fun getValue(vararg weightedInputs: Double): Double

    /**
     * A neuron that fires an input value.
     */
    class InputNeuron(val inputFunction: () -> Double) : Neuron {
        override fun getValue(vararg weightedInputs: Double) = inputFunction()
    }

    /**
     * A neuron that fires depending on an activation function.
     */
    class ActivationNeuron(val activation: ActivationFunction) : Neuron {
        override fun getValue(vararg weightedInputs: Double) = activation.eval(weightedInputs.sum())
    }

    /**
     * A neuron that always fires with a value of 1.
     */
    class BiasNeuron : Neuron {
        override fun getValue(vararg weightedInputs: Double) = 1.0
    }
}
