package net.benwoodworth.neuralnetworks.network

/**
 * Provides options for mutation of a neural network.
 */
class NetworkMutator(val weightMutation: MutateOffset,
                     val activateMutation: MutateBool) {

    fun mutateWeight(original: Double) = original + weightMutation.getRandChange()

    fun mutateActivation(original: ActivationFunction): ActivationFunction {
        if (!activateMutation.getRandBool()) return original

        val functions = ActivationFunction.AllFunctions
        return functions[(Math.random() * functions.size).toInt()]
    }

    fun mutateNeuron(neuron: Neuron): Neuron = when (neuron) {
        is Neuron.ActivationNeuron -> Neuron.ActivationNeuron(mutateActivation(neuron.activation))
        else -> neuron
    }

    class MutateOffset(val changeProbability: Double, val offsetMin: Double, val offsetMax: Double) {
        fun getRandChange(): Double {
            return if (Math.random() < changeProbability)
                Math.random() * (offsetMax - offsetMin) + offsetMin
            else
                0.0
        }
    }

    class MutateBool(val changeProbability: Double) {
        fun getRandBool() = Math.random() < changeProbability
    }
}
