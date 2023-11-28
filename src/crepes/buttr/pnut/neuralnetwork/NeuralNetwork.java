package crepes.buttr.pnut.neuralnetwork;

import java.util.ArrayList;

public class NeuralNetwork {

	private ArrayList<ArrayList<Neuron>> neurons;
	private ArrayList<ArrayList<NeuralConnection>> connections;
	
	private int[] layerSizes;
	
	private double[] expectedValues;
	
	protected double learningSpeed;
	
	protected NetworkMath.ActivationFunction hiddenActivationFunction;
	protected NetworkMath.ActivationFunction outputActivationFunction;
	
	/**
	 * 
	 * @param layerSizes - an integer array containing the number
	 * of neurons to be generated for each layer of the neural
	 * network. The first element in the array is the number
	 * of input neurons the neural network will have, the last element
	 * in the array is the number of output neurons, and any
	 * numbers in between correspond to the hidden layers of the
	 * neural network.
	 * @param zeroed - if <code> true </code>, the weights of
	 * all of the neurons within the network will be automatically
	 * set to 0, and if <code> false </code>, the weights of
	 * the neurons with be generated randomly.
	 */
	public NeuralNetwork(int[] layerSizes, boolean zeroed) {
		
		for(int i = 0; i < layerSizes.length; i++) {
			
			if(layerSizes[i] < 1)
				layerSizes[i] = 1;
		}
		
		if(layerSizes.length > 1)
			this.layerSizes = layerSizes;
		else
			this.layerSizes = new int[]{1, 1};
		
		neurons = new ArrayList<ArrayList<Neuron>>();
		connections = new ArrayList<ArrayList<NeuralConnection>>();
		
		for(int i = 0; i < this.layerSizes.length; i++) {
			
			neurons.add(new ArrayList<Neuron>());
			
			for(int j = 0; j < layerSizes[i]; j++) {
				
			    if(j != layerSizes[i] - 1)
				neurons.get(i).add(new Neuron(hiddenActivationFunction));
			    else
				neurons.get(i).add(new Neuron(outputActivationFunction));
			}
		}
		
		for(int i = 0; i < neurons.size() - 1; i++) {
			
			connections.add(new ArrayList<NeuralConnection>());
			
			for(Neuron previousNeuron : neurons.get(i)) {
				for(Neuron nextNeuron : neurons.get(i + 1)) {
					
					connections.get(i).add(new NeuralConnection(previousNeuron, nextNeuron, zeroed, hiddenActivationFunction));
				}
			}
		}
		
		this.expectedValues = new double[neurons.get(neurons.size() - 1).size()];
		
		hiddenActivationFunction = NetworkMath.ActivationFunction.RELU;
		outputActivationFunction = NetworkMath.ActivationFunction.SIGMOID;
	}
	
	/**
	 * Sets the activation values of the neurons of the first
	 * layer of the neural network. Note: this command DOES
	 * NOT tell the network to forward propagate. To tell the
	 * network to propagate these values, call the
	 * <code> forwardPropagate() </code> method of the network.
	 * @see #forwardPropagate()
	 * @param inputValues - all values to be propagated forward
	 * by the network.
	 */
	public void passInputs(double[] inputValues) {

		for(int i = 0; i < neurons.get(0).size(); i++) {
				
			if(i < inputValues.length)
				neurons.get(0).get(i).setActivation(inputValues[i]);
			else
				neurons.get(0).get(i).setActivation(0);
		}
	}
	
	/**
	 * Tells the network to produce outputs by propagating the
	 * input values given to it by the <code> passInputs()
	 * </code> method forward.
	 * @see #passInputs(double[])
	 */
	public void forwardPropagate() {
		
		for(int i = 1; i < neurons.size(); i++) {
			for(Neuron neuron : neurons.get(i)) {
				
				neuron.setActivation(0);
			}
		}
		
		for(int i = 0; i < connections.size(); i++) {
			
			for(NeuralConnection connection : connections.get(i)) {
			
				connection.forwardPropagate();
			}
			
			for(Neuron neuron : neurons.get(i+1)) {
				
				neuron.addBias();
				neuron.performActivationFunction();
			}
		}
	}
	
	/**
	 * Returns the values of the output neurons of the
	 * neural network.
	 * @return An array containing the values of the last
	 * layer of neurons.
	 */
	public double[] getOutputs() {
		
		double[] outputValues = new double[neurons.get(neurons.size() - 1).size()];
		
		for(int i = 0; i < neurons.get(neurons.size() - 1).size(); i++) {
			
			outputValues[i] = neurons.get(neurons.size() - 1).get(i).getActivation();
		}
		
		return outputValues;
	}
	
	/**
	 * Returns the values of the output neurons of the
	 * neural network, rounded to the nearest integer.
	 * @return An array containing the values of the last
	 * layer of neurons, rounded to the nearest integer.
	 */
	public int[] getRoundedOutputs() {
		
		int[] outputValues = new int[neurons.get(neurons.size() - 1).size()];
		
		for(int i = 0; i < neurons.get(neurons.size() - 1).size(); i++) {
			
			outputValues[i] = (int) Math.round(neurons.get(neurons.size() - 1).get(i).getActivation());
		}
		
		return outputValues;
	}
	
	/**
	 * This function receives the values against which the
	 * neural network should compare its own outputs. These
	 * are the values used for calculating the cost and average
	 * cost of the neural network, as well as the target values
	 * used to calculate derivatives during backpropagation.
	 * @param expectedValues - an array containing of the
	 * correct values for all of the output neurons.
	 */
	public void passCorrectOutputValues(double[] expectedValues) {
		
		for(int i = 0; i < this.expectedValues.length; i++) {
			
			if(i < expectedValues.length)
				this.expectedValues[i] = expectedValues[i];
			else
				this.expectedValues[i] = 0;
		}
		
		computeSquareCost();
	}
	
	/**
	 * Returns the value of the cost function of the network,
	 * which is computed as the sum of the squares of the
	 * differences between the predicted output values and the
	 * actual output values.
	 * @return A floating point number with a value within
	 * the allowed range for floating point numbers.
	 */
	public double computeSquareCost() {
		
		//TODO Don't have it compute cost from scratch every time.
		
		double cost = 0;
		
		double[] outputs = getOutputs();
		
		for(int i = 0; i < outputs.length; i++) {
			
			cost += Math.pow(expectedValues[i] - outputs[i], 2);
		}
		
		return cost;
	}
	
	//TODO Implement average square cost function, accrued cost divided by a number of trials
	public double computeAverageSquareCost() {
		
		
		
		return 0;
	}
	
	/**
	 * Tells the network to adjust its weights and biases
	 * using the correct output value given to it by the
	 * <code> passCorrectOutputValues </code> method
	 * by calculating the derivative of the cost with
	 * respect to each weight and bias and subtracting the
	 * derivative, multiplied by the learning speed, from
	 * the current value of the weight or bias. This brings
	 * the output of the network closer to the desired
	 * output.
	 * @see #passCorrectOutputValues(double[])
	 */
	public void backPropagate() {
		
		for(int i = 0; i < neurons.size(); i++) {
			for(int j = 0; j < neurons.get(i).size(); j++) {
			
				neurons.get(i).get(j).setCostDerivative(0);
			}
		}
		
		for(int i = 0; i < neurons.get(neurons.size() - 1).size(); i++) {
			
			neurons.get(neurons.size() - 1).get(i).calculateOutputLayerDerivative(expectedValues[i], learningSpeed);
		}
		
		for(int i = connections.size() - 1; i >= 0; i--) {
			for(int j = 0; j < connections.get(i).size(); j++) {
				
				connections.get(i).get(j).backPropagate();
			}
		}
	}
	
	public void setActivationFunctions(NetworkMath.ActivationFunction hiddenActivationFunction, NetworkMath.ActivationFunction outputActivationFunction)
	{
	    this.hiddenActivationFunction = hiddenActivationFunction;
	    this.outputActivationFunction = outputActivationFunction;
	}
	
	/**
	 * This functions sets the learning speed of the neural
	 * network in micro-units.
	 * @param microLearningSpeed - The learning speed of the
	 * neural network in micro-units (for example, passing a 1
	 * will produce a learning speed of 0.000001). This value is
	 * multiplied by the calculated derivatives to determine
	 * the necessary change in each value associated with the
	 * neural network.
	 */
	public void setMilliLearningSpeed(double microLearningSpeed) {
		
		this.learningSpeed = microLearningSpeed * 0.001;
	}
	
	public double getNeuronBias(int layer, int index) {
		
		return neurons.get(layer).get(index).getBias();
	}
	
	public void setNeuronBias(int layer, int index, double bias) {
		
		neurons.get(layer).get(index).setBias(bias);
	}
	
	public double getConnectionWeight(int layer, int index) {
		
		return connections.get(layer).get(index).getWeight();
	}
	
	public void setConnectionWeight(int layer, int index, double weight) {
		
		connections.get(layer).get(index).setWeight(weight);
	}
	
	public ArrayList<ArrayList<Neuron>> getNeuronLayers() {
		
		return neurons;
	}
	
	public ArrayList<ArrayList<NeuralConnection>> getNeuralConnectionLayers() {
		
		return connections;
	}
}
