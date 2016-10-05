package net.benwoodworth.neuralnetworks.network

/**
 * Interface for providing input functions no a neural network.
 */
interface NeuralInputProvider {
    fun getNeuralInputs(): () -> Double
}
