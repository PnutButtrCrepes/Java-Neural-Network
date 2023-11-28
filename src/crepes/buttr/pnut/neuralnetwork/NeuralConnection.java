package crepes.buttr.pnut.neuralnetwork;

public class NeuralConnection {

	private Neuron previousNeuron;
	private Neuron nextNeuron;
	
	private double weight;
	
	private NetworkMath.ActivationFunction activationFunction;
	
	protected NeuralConnection(Neuron previousNeuron, Neuron nextNeuron, boolean zeroed, NetworkMath.ActivationFunction activationFunction) {
		
		this.previousNeuron = previousNeuron;
		this.nextNeuron = nextNeuron;
		
		if(zeroed)
			this.weight = 0;
		else
			this.weight = Math.random() - 0.5;
		
		this.activationFunction = activationFunction;
	}
	
	protected void forwardPropagate() {
		
		nextNeuron.setActivation(nextNeuron.getActivation() + previousNeuron.getActivation() * weight);
	}
	
	protected void backPropagate() {
		
	    switch(activationFunction)
	    {
	    case RELU:
		
		previousNeuron.setCostDerivative(previousNeuron.getCostDerivative() + nextNeuron.getCostDerivative() * weight *
			NetworkMath.reluPrime(previousNeuron.getActivation()));
		
		break;
		
	    case SIGMOID:
		
		previousNeuron.setCostDerivative(previousNeuron.getCostDerivative() + nextNeuron.getCostDerivative() * weight *
			NetworkMath.sigmoidPrime(previousNeuron.getActivation()));
		
		break;
		
	    case SOFTMAX:
		break;
		
	    default:
		break;
	    }
		
		previousNeuron.setBias(previousNeuron.getBias() - previousNeuron.getCostDerivative());
		
		weight -= nextNeuron.getCostDerivative() * previousNeuron.getActivation();
	}

	protected double getWeight() {
		
		return weight;
	}

	protected void setWeight(double weight) {
		
		this.weight = weight;
	}
}