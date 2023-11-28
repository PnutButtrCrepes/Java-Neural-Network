package crepes.buttr.pnut.deepqlearning;

import java.util.ArrayList;

import crepes.buttr.pnut.neuralnetwork.*;

/**
 * This is a class that models a Deep Q-Learning Network
 * in java.
 * @author Nathan Keenan
 */

public class DeepQLearningNetwork {

	private NeuralNetwork dynamicNetwork;
	private NeuralNetwork targetNetwork;
	
	private double epsilon;
	
	private double confidence;
	private double discountRate;
	
	private double[] state;
	private int lastAction;
	private double lastReward;
	private boolean terminal;
	private double[] statePrime;
	
	private ArrayList<Experience> replayBuffer;
	private int maxReplayBufferSize;
	private int replayBufferRefreshRate;
	private int numberOfReplayBufferSamples;
	private int networkRefreshRate;
	
	private double prioritization;
	
	private long numberOfDecisions;
	private int numberOfBatchSamplesSinceCopy;
	
	/**
	 * Network constructor which takes the desired size of the
	 * network as an array, where each index in the array
	 * represents the number of neurons in that layer of the
	 * network.
	 * @param layerSizes - an array representing the size of
	 * each layer of the neural network.
	 */
	public DeepQLearningNetwork(int[] layerSizes) {
		
		this.dynamicNetwork = new NeuralNetwork(layerSizes, false);
		this.targetNetwork = new NeuralNetwork(layerSizes, true);
		
		replayBuffer = new ArrayList<Experience>();
		numberOfDecisions = 0;
		numberOfBatchSamplesSinceCopy = 0;
	}
	
	/**
	 * Returns the index of the output neuron that represents
	 * the appropriate action to be taken by the agent given
	 * the current state, the agent being the entity controlled
	 * by the network. Note: only one action can be taken at time.
	 * Therefore, to implement an action that consists of
	 * two or more sub-actions (for example, an agent learning
	 * to play a video game that may require more than one button
	 * to be pressed at a time), implement the combination
	 * thereof as a separate action entirely (pressing one of the
	 * buttons separately is one action, pressing the other button
	 * separately is a second action, and pressing both together
	 * is a third action).
	 * @param state - This is an array providing a numerical
	 * representation of the agent's environment. These are also
	 * the values that will be passed to the neural network to
	 * determine the agent's next action.
	 * @return An integer with a value no less than 0 and less
	 * than the number of output neurons in the network.
	 */
	public int getEpsilonGreedyActionIndex(double[] state) {
		
		this.state = state;
		
		dynamicNetwork.passInputs(state);
		dynamicNetwork.forwardPropagate();
		
		epsilon = calculateEpsilon();
		double random = Math.random();
		
		if(random < epsilon) {
			
			lastAction = (int) (Math.random() * dynamicNetwork.getNeuronLayers().get(dynamicNetwork.getNeuronLayers().size() - 1).size());
			
		} else {
		
			double[] outputs = dynamicNetwork.getOutputs();
			
			lastAction = getIndexOfHighestQValue(outputs);
		}
		
		numberOfDecisions++;
		
		return lastAction;
	}
	
	private double calculateEpsilon() {
		
		return (1 / Math.pow((numberOfDecisions + 1), confidence));
	}
	
	private int getIndexOfHighestQValue(double[] outputs) {
		
		int indexOfLargestValue = 0;
		int[] equalActions = new int[outputs.length];
		int numberOfEqualActions = 0;
		
		for(int i = 1; i < outputs.length; i++) {
			
			if(outputs[i] > outputs[indexOfLargestValue]) {
				
				indexOfLargestValue = i;
				
				numberOfEqualActions = 0;
				
			} else if(outputs[i] == outputs[i - 1]) {
				
				if(numberOfEqualActions == 0) {
					
					equalActions[numberOfEqualActions] = i - 1;
					equalActions[numberOfEqualActions + 1] = i;
					numberOfEqualActions += 2;
					
				} else {
					
					equalActions[numberOfEqualActions] = i;
					numberOfEqualActions++;
				}
			}
		}
		
		if(numberOfEqualActions > 1) {
			
			double random = Math.random();
			
			Action:
			for(int i = 0; i < numberOfEqualActions; i++) {
				
				if(random < (i + 1) / numberOfEqualActions) {
					
					indexOfLargestValue = equalActions[i];
					break Action;
				}
			}
		}
		
		return indexOfLargestValue;
	}
	
	/**
	 * Updates the reward and the resultant state for the action
	 * taken in the previous state. Note: this method should be
	 * called on the frame immediately following the frame on which
	 * the action was taken, but before any other action is taken.
	 * Furthermore, this method DOES NOT tell the network to back
	 * propagate, it merely tells the network to store the state,
	 * action, reward, and statePrime tuple in the replay buffer.
	 * @param reward - the reward given to the agent for taken
	 * a specific action on the previous frame.
	 * @param statePrime - the state that resulted from taking the
	 * previous action in the previous state. These are the same
	 * values that will be passed to the network as input this frame.
	 */
	public void passRewardAndStatePrime(double reward, boolean terminal, double[] statePrime) {
		
		this.lastReward = reward;
		this.terminal = terminal;
		this.statePrime = statePrime;
		
		if(maxReplayBufferSize > 0) {
			
			if(replayBuffer.size() == maxReplayBufferSize) {
				
				replayBuffer.remove(0);
			}
			
			//TODO the parameter prioritization should probably be used during playback, but what if we apply it to memory formation below?
			
			if(lastReward == 0)
			{
			    if(Math.random() >= prioritization)
				replayBuffer.add(new Experience(this.state, this.lastAction, this.lastReward, this.terminal, this.statePrime));
			}
			else
			{
			    replayBuffer.add(new Experience(this.state, this.lastAction, this.lastReward, this.terminal, this.statePrime));
			}
		}
	}
	
	/**
	 * 
	 */
	
	public void checkForBatchRefresh() {
		
		if(numberOfDecisions % replayBufferRefreshRate == 0 && replayBuffer.size() > 0) {
			
			int sampleIndex;
			Experience sample;
			double[] correctedOutputs;
			
			for(int i = 0; i < numberOfReplayBufferSamples; i++) {
				
				sampleIndex = (int) (Math.random() * replayBuffer.size());
				sample = replayBuffer.get(sampleIndex);
				
				dynamicNetwork.passInputs(sample.getState());
				dynamicNetwork.forwardPropagate();
				
				targetNetwork.passInputs(sample.getStatePrime());
				targetNetwork.forwardPropagate();
				
				correctedOutputs = dynamicNetwork.getOutputs();
				
				if(!sample.getTerminal()) {
					
				correctedOutputs[sample.getActionIndex()] = sample.getReward() +
						discountRate * targetNetwork.getOutputs()[getIndexOfHighestQValue(targetNetwork.getOutputs())];
				
				} else {
					
					correctedOutputs[sample.getActionIndex()] = sample.getReward();
				}
				
				
				dynamicNetwork.passCorrectOutputValues(correctedOutputs);
				dynamicNetwork.backPropagate();
				
				numberOfBatchSamplesSinceCopy++;
				checkForTargetNetworkRefresh();
			}
		}
	}
	
	private void checkForTargetNetworkRefresh() {
		
		if(numberOfBatchSamplesSinceCopy == networkRefreshRate) {
			
			for(int i = 0; i < targetNetwork.getNeuronLayers().size(); i++) {
				for(int j = 0; j < targetNetwork.getNeuronLayers().get(i).size(); j++) {
					
					targetNetwork.setNeuronBias(i, j, dynamicNetwork.getNeuronBias(i, j));
				}
			}
			
			for(int i = 0; i < targetNetwork.getNeuralConnectionLayers().size(); i++) {
				for(int j = 0; j < targetNetwork.getNeuralConnectionLayers().get(i).size(); j++) {
					
					targetNetwork.setConnectionWeight(i, j, dynamicNetwork.getConnectionWeight(i, j));
				}
			}
			
			numberOfBatchSamplesSinceCopy = 0;
		}
	}
	
	/**
	 * Returns the value of the cost function of the network,
	 * which is computed as the sum of the squares of the
	 * differences between the predicted Q-Values and the
	 * actual Q-Values.
	 * @return A floating point number with a value within
	 * the allowed range for floating point numbers.
	 */
	public double computeSquareCost() {
		
		return dynamicNetwork.computeSquareCost();
	}
	
	/**
	 * Sets all relevant hyperparameters of the network.
	 * @param confidence - influences the probability that the
	 * network will pick the best option over a random option.
	 * The higher the value, the more likely the network will
	 * be to choose the best option. The chance of choosing a
	 * random option is calculated as follows:
	 * <code> 1 / Math.pow((numberOfActionsTaken + 1), confidence)
	 * </code>
	 * @param discountRate - a parameter that tells the network
	 * how much to prioritize future rewards over immediate
	 * rewards, typically having a value between 0 and 1 inclusive
	 * (usually 0.9). The higher the value, the more the network
	 * will value future rewards over immediate rewards.
	 * @param milliLearningSpeed - a constant that controls how much
	 * the weights and biases are changed after the network propagates
	 * backward. The value must be given in milliunits, thus an input
	 * of 1 is really 0.001. This is because the learning speed should
	 * range between 0.000001 and 1.
	 * @param networkRefreshRate - the number of actions taken after
	 * which the target network is updated to match the
	 * dynamic network.
	 * @param maxReplayBufferSize - the maximum number of
	 * experiences that the network stores in memory.
	 * @param replayBufferRefreshRate - the number of decisions
	 * that the network must make before the network is trained
	 * using the replay buffer.
	 * @param numberOfReplayBufferSamples - the number of
	 * experiences the network cycles through per training
	 * session. An experience is made up of a state, the action
	 * taken, the resultant reward, and the resultant state.
	 * @param prioritization - a parameter that tells the network how
	 * much it should prioritize learning from unusual experiences.
	 * That is, experiences in the replay buffer for which the cost
	 * was calculated to be higher. A value of 0 means that the
	 * network selects the experiences that it learns from randomly.
	 */
	public void setNetworkHyperParameters(double confidence, double discountRate, double milliLearningSpeed,
			int maxReplayBufferSize, int replayBufferRefreshRate, int numberOfReplayBufferSamples, int networkRefreshRate,
			double prioritization) {
		
		this.confidence = confidence;
		this.discountRate = discountRate;
		dynamicNetwork.setMilliLearningSpeed(milliLearningSpeed);
		
		this.maxReplayBufferSize = maxReplayBufferSize;
		this.replayBufferRefreshRate = replayBufferRefreshRate;
		this.numberOfReplayBufferSamples = numberOfReplayBufferSamples;
		this.networkRefreshRate = networkRefreshRate;
		
		this.prioritization = prioritization;
	}
	
	public double[] getOutputs() {
		
		return dynamicNetwork.getOutputs();
	}
}