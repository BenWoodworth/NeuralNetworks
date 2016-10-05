package net.benwoodworth.neuralnetworks.network

/**
 * A network of neurons that calculates outputs given inputs.
 */
class NeuralNetwork {

    /**
     * The weights between neurons in all consecutive layers.
     * Usage: `weights[from layer][from neuron][to next layer neuron]`
     */
    private val weights: List<List<List<Double>>>

    /**
     * Each layer is a list of neurons.
     */
    val layers: List<List<Neuron>>

    constructor(initialWeights: () -> Double, vararg layers: List<Neuron>) {
        // Set all weights to initialWeights
        weights = (layers.indices - layers.indices.last).map {
            (0..layers[it].size).map {
                (0..layers[it + 1].size).map { initialWeights() }
            }
        }

        this.layers = layers.asList()
    }

    constructor(initialWeights: Double, vararg layers: List<Neuron>) : this({initialWeights}, *layers)

    constructor(parent: NeuralNetwork, mutator: NetworkMutator) {
        weights = parent.weights.map {
            it.map { it.map { mutator.mutateWeight(it) } }
        }

        layers = parent.layers.map {
            it.map { mutator.mutateNeuron(it) }
        }
    }

    fun calculateOutputs(): Array<Double> {
        // Initialize with input neuron values
        var prevLayerOutputs = Array(layers[0].size) {
            layers[0][0].getValue()
        }

        // Sequentially replace prevLayerOutputs with the outputs of the current layer
        for (iCurLayer in layers.indices - 0) {
            val curLayer = layers[iCurLayer]
            prevLayerOutputs = Array(curLayer.size) {
                // Evaluate the current neuron, and add the result to the outputs
                curLayer[it].getValue(prevLayerOutputs.indices.map {
                    // Multiply output from the previous layer with the connection weight
                    val prevLayerOutput = prevLayerOutputs[it]
                    val connectionWeight = weights[iCurLayer - 1][it][iCurLayer]
                    prevLayerOutput * connectionWeight
                }.sum())
            }
        }

        return prevLayerOutputs
    }
}
