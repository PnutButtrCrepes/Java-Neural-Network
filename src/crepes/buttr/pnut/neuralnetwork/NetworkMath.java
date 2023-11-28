package crepes.buttr.pnut.neuralnetwork;

public class NetworkMath {

	private static double negativeReLUGradient = 0.01;
	
	protected static double relu(double activation) {
		
		if(activation < 0)
		    return activation * negativeReLUGradient;
		else
		    return activation;
	}
	
	protected static double reluPrime(double activation) {
		
		if(activation < 0)
		    return negativeReLUGradient;
		else
		    return 1;
	}
	
	protected static double sigmoid(double activation) {
		
		return 1 / (1 + Math.pow(Math.E, -activation));
	}
	
	protected static double sigmoidPrime(double activation) {
		
		return sigmoid(activation) * (1 - sigmoid(activation));
	}
	
	//TODO need to add softmax functions here
	
	protected static double getNegativeGradient() {
		
		return negativeReLUGradient;
	}
	
	protected static void setNegativeGradient(double newNegativeGradient) {
		
	    	negativeReLUGradient = newNegativeGradient;
	}
	
	public enum ActivationFunction {
	    
	    RELU,
	    SIGMOID,
	    SOFTMAX
	}
}