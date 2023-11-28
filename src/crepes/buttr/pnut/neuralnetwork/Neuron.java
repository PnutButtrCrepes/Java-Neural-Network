package crepes.buttr.pnut.neuralnetwork;

public class Neuron {
	
	private double activation;
	
	private double bias;
	
	private double costDerivative;
	
	private NetworkMath.ActivationFunction activationFunction;

	protected Neuron(NetworkMath.ActivationFunction activationFunction) {
		
		this.activation = 0;
		
		this.bias = 0;
		
		this.activationFunction = activationFunction;
	}
	
	protected void calculateOutputLayerDerivative(double expectedValue, double learningSpeed) {
		
	    switch(activationFunction)
	    {
	    case RELU:
		
		costDerivative = 2 * (activation - expectedValue) * NetworkMath.reluPrime(activation) * learningSpeed;
		
		break;
		
	    case SIGMOID:
		
		costDerivative = 2 * (activation - expectedValue) * NetworkMath.sigmoidPrime(activation) * learningSpeed;
		
		break;
		
	    case SOFTMAX:
		break;
		
	    default:
		break;
	    }
		
		bias -= costDerivative;
	}
	
	protected void addBias() {
		
		activation = activation + bias;
	}
	
	protected void performActivationFunction() {
			
	    switch(activationFunction)
	    {
	    case RELU:
		
		this.activation = NetworkMath.relu(activation);
		
		break;
		
	    case SIGMOID:
		
		this.activation = NetworkMath.sigmoid(activation);
		
		break;
		
	    case SOFTMAX:
		break;
		
	    default:
		break;
	    }
	}
	
	protected double getActivation() {
		
		return activation;
	}

	protected void setActivation(double activation) {
		
		this.activation = activation;
	}

	protected double getBias() {
		
		return bias;
	}

	protected void setBias(double bias) {
		
		this.bias = bias;
	}
	
	protected double getCostDerivative() {
		
		return costDerivative;
	}

	protected void setCostDerivative(double costDerivative) {
		
		this.costDerivative = costDerivative;
	}
}
