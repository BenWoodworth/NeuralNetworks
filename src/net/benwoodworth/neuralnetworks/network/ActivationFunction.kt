package net.benwoodworth.neuralnetworks.network

import java.lang.Math.*

/**
 * Activation function, for use by neurons.
 */
interface ActivationFunction {

    /**
     * Evaluate the activation function.
     */
    fun eval(x: Double): Double

    /**
     * Evaluate the derivative of the activation function.
     */
    fun deriv(x: Double): Double

    /**
     * Activation functions
     */
    companion object {
        val Identity = object : ActivationFunction {
            override fun eval(x: Double) = x
            override fun deriv(x: Double) = 1.0
        }

        val Step = object : ActivationFunction {
            override fun eval(x: Double) = if (x >= 0) 1.0 else 0.0
            override fun deriv(x: Double) = 0.0
        }

        val TanH = object : ActivationFunction {
            override fun eval(x: Double) = 2 / (1 + exp(-2 * x)) - 1
            override fun deriv(x: Double) = 1 - eval(x) * eval(x)
        }

        val ATan = object : ActivationFunction {
            override fun eval(x: Double) = atan(x)
            override fun deriv(x: Double) =  1 / (x * x + 1)
        }

        val Rectifier = object : ActivationFunction {
            override fun eval(x: Double) = if (x >= 0) x else 0.0
            override fun deriv(x: Double) = if (x >= 0) 1.0 else 0.0
        }

        val AllFunctions = listOf(Identity, Step, TanH, ATan, Rectifier)
    }
}
