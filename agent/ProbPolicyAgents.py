from agent.AgentBase import AgentBase, Verbosity
from ParkingDefs import StateType, Act
from random import random

class OccupiedRandomAgent(AgentBase):
    def __init__(self, name, numSpacesPerRow, parkProb, verbosity):
        super().__init__(name, verbosity)
        self.parkProb = parkProb
        self.numSpacesPerRow = numSpacesPerRow # Need this for output reasons

    def selectAction(self, state, iteration, mdp):
        stateClass = StateType.get(state)
        # check to see that the space is not occupied by a car
        if StateType.DRIVING_AVAILABLE == stateClass:
            if random() < self.parkProb:
                return Act.PARK.value
        return Act.DRIVE.value

    # Override some printing functions so we can correctly figure out state semantics
    def observeReward(self, iteration, currentState, nextState, action, totalReward, rewardHere):
        if self.verbosity.value <= Verbosity.REWARDS.value:
            print("*************************")
            print("t : " + str(iteration) + "   Total Reward : \t" + str(totalReward))
            print("Action       : \t" + str(Act(action)))
            print("Reward       : \t", rewardHere)
            print("Current State: \t" + StateType.interpretState(currentState, self.numSpacesPerRow))
    def endEpisode(self, iteration, finalState, totalReward):
        if self.verbosity.value <= Verbosity.RESULTS.value:
            print("***************************************************************************")
            print("Iterations used       : \t" + str(iteration + 1))
            print("Final State: \t" + StateType.interpretState(finalState, self.numSpacesPerRow))
            print("!!!Reward for this trial: \t", totalReward)



class OccupiedRandomNoHandicapAgent(OccupiedRandomAgent):
    def __init__(self, name, numSpacesPerRow, parkProb, verbosity):
        super().__init__(name, numSpacesPerRow, parkProb, verbosity)

    def selectAction(self, state, iteration, mdp):
        stateClass = StateType.get(state)
        if StateType.DRIVING_AVAILABLE == stateClass:
            # state-3 is the "parked" state associated with this space.
            # By checking that the reward is positive, we ensure we do not park in handicapped spaces.
            if mdp.rewardFn[state-3] > 0:
                if random() < self.parkProb:
                    return Act.PARK.value
        return Act.DRIVE.value



class OccupiedRandomNoHandicapLapAgent(OccupiedRandomAgent):
    def __init__(self, name, numSpacesPerRow, parkProb, lapFactor, verbosity):
        super().__init__(name, numSpacesPerRow, parkProb, verbosity)
        self.lapFactor = lapFactor

    def selectAction(self, state, iteration, mdp):
        stateClass = StateType.get(state)
        if StateType.DRIVING_AVAILABLE == stateClass:
            if mdp.rewardFn[state-3] > 0:
                # This check will make the agent "impatient" after completing some number of laps in the parking lot.
                # After completing that amount, it will always greedily park in the next available space
                if StateType.lapFinished(iteration, mdp.numStates, self.lapFactor) or random() < self.parkProb:
                    return Act.PARK.value
        return Act.DRIVE.value

class YourAgent(OccupiedRandomAgent):
    def __init__(self, name, numSpacesPerRow, parkProb, lapFactor, verbosity):
        super().__init__(name, numSpacesPerRow, parkProb, verbosity)
        self.lapFactor = lapFactor

    def selectAction(self, state, iteration, mdp):

        def adjustParkProbability(state, iteration, mdp):
            # Example of a simple strategy: increase parking probability with each lap completed
            lapsCompleted = iteration // mdp.numStates
            adjustedProb = min(self.parkProb + lapsCompleted * 0.1, 1.0)  # Ensure probability doesn't exceed 1
            return adjustedProb

        stateClass = StateType.get(state)

        # Adjust park probability dynamically based on conditions
        dynamicParkProb = adjustParkProbability(state, iteration, mdp)

        if StateType.DRIVING_AVAILABLE == stateClass:
            # Avoid parking in handicapped spaces (assuming negative reward indicates such spaces)
            if mdp.rewardFn[state-3] > 0:
                # Increase urgency to park based on the number of laps completed
                if StateType.lapFinished(iteration, mdp.numStates, self.lapFactor) or random() < dynamicParkProb:
                    return Act.PARK.value
        return Act.DRIVE.value