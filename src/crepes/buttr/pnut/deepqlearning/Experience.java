package crepes.buttr.pnut.deepqlearning;

public class Experience {

	private double[] state;
	private int actionIndex;
	private double reward;
	private boolean terminal;
	private double[] statePrime;
	
	protected Experience(double[] state, int actionIndex, double reward, boolean terminal, double[] statePrime) {
		
		this.state = state;
		this.actionIndex = actionIndex;
		this.reward = reward;
		this.statePrime = statePrime;
	}

	protected double[] getState() {
		
		return state;
	}

	protected int getActionIndex() {
		
		return actionIndex;
	}

	protected double getReward() {
		
		return reward;
	}

	protected boolean getTerminal() {
		
		return terminal;
	}
	
	protected double[] getStatePrime() {
		
		return statePrime;
	}
}