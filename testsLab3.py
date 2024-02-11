from random import seed
from ParkingLotFactory import createParkingMDP
from agent.QLearningAgent import QLearningAgent
from MDP import MDP
from agent.AgentBase import Verbosity
from statistics import stdev

numRows = 2
numRegularSpacesPerRow = 10
numHandicappedSpacesPerRow = 5
busyRate = .3
handicapBusyRate = .05
parkedReward = 1000
crashPenalty = -10000
waitingPenalty = -1
decayBusyRate = False
decayReward = True

numActions = 2
parkProb = .5
numSpacesPerRow = numRegularSpacesPerRow + numHandicappedSpacesPerRow
start = 4 * numSpacesPerRow - 2 # Be careful to update this variable and the previous one if you change the parking lot size
discountFactor = .7
learningRate = .1
probGreedy = .9


# A helper function we will use often
def printRewards(epoch, rewardList, agentName, mdpName):
    '''
    This function prints the rewards from each epoch.
    '''
    print("At Epoch\t{}\tOn\t{}\t, Over {} Trajectories, Agent\t{}\t achieved a Grand Total Reward:\t{:.4f}\t, for an average of\t{:.4f}\t with std\t {:.4f}\t.".format(
            epoch, mdpName, len(rewardList), agentName, sum(rewardList), sum(rewardList) / len(rewardList),
            stdev(rewardList)), end="\t")
    ## Uncomment the below lines if you want to print the list and not stick it in a file
    # for i in range(len(rewardList)):
    #     print(rewardList[i], end="\t")
    print()


def trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent):
    '''
    This is the function to train the Q-learning table
    The reward list will be stored in {agentName}_{mdp}_rewardList.txt:
        where each line represent the rewardlist for the corresponding epoch
    The reward for each epoch will be stored in {agentName}_{mdp}_rewardCurveData.txt
        where each line represent the average reward of each simulation sample for the corresponding epoch
    Parameters
    ----------
    numEpochs: int
        number of Epochs to train the Q-learning table
    numSamples: int
        number of samples to simulate in each epoch to update the Q-learning table
    agent: QLearningAgent
        The agent to be trained
    '''
    # create the files for rewardList and rewardCurveData
    rewardList_f = open("output/{}_{}_rewardList.txt".format(agent.name, mdp.name), "w")
    rewardCurve_f = open("output/{}_{}_rewardCurveData.txt".format(agent.name, mdp.name), "w")

    for epoch in range(numEpochs):
        # first, learn for <numSamples> trajectories
        for trajectory in range(numSamples):
            agent.setLearningRate(trajectory, numSamples)
            _ = mdp.simulateTrajectory(agent, startState=start)
            if agent.verbosity == Verbosity.VERBOSE:
                print("\n **************************Epoch {}, sample trial: {} Completed**************************".format(epoch + 1, trajectory + 1))

        # second, evaluate for <numSamples> trajectories
        agent.evaluating = True
        rewardList = []
        for _ in range(numSamples):
            reward = mdp.simulateTrajectory(agent, startState=start)
            rewardList.append(reward)
        agent.evaluating = False

        # last, print to console and write to files
        printRewards(epoch, rewardList, agent.name, mdp.name)
        # input the data into {agentName}_{mdp}_rewardList.txt and {agentName}_{mdp}_rewardCurveData.txt
        rewardList_f.write(str.join("\t", [str(reward) for reward in rewardList]) + "\n")
        rewardCurve_f.write(str(sum(rewardList) / len(rewardList)) + "\n")

    rewardList_f.close()
    rewardCurve_f.close()


############ Task1: Understand updates on a Q-value table ###########
# In this task, we will apply the Q-learning on the simple MDP we saw in Lab1 (on slides and in MDP1.txt)
# Follow the detailed description in CANVAS to complete Task1
def lab3test1(seedValue):
    '''
    The Q-value table of all states with different actions will be printed for each sample trial in each example
    Average rewards of each trail will be reported
    This function will create the following files:
        BasicQLearner_Lab 1's toy MDP_rewardCurveData.txt: Average reward of each epoch
        BasicQLearner_Lab 1's toy MDP_rewardList.txt: Reward List of each epoch
    '''
    print("\n\n\n----------------------- BEGIN Test 1 - Q-Learning on the first toy MDP we saw in Lab 1 (the movie), with Seed " + str(seedValue))
    seed(seedValue)

    # setup the mdp parameters
    start = 0
    mdp = MDP(None, None, "MDP1", "data/MDP1.txt")

    # set amount of training/testing to something small (so we arent flooded with output)
    numSamples = 2
    numEpochs = 1
    agent = QLearningAgent("BasicQLearner", numActions, mdp.numStates, probGreedy, discountFactor, learningRate,
                           Verbosity.VERBOSE)
    trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent)


############ **Task2: Creating learning curve of the Q-learning agent on the parking MDP** ###########
# Learning curve is a line chart which x-axis is the epoch and y-axis is the average rewords
# Your will run test2 and test3 to complete task 2 by following the detailed description in CANVAS
def lab3test2(seedValue):
    '''
    This is a test on the Small Parking MDP
    This function will create the following files:
        BasicQLearner_Small Parking MDP_rewardCurveData.txt: Average reward of each epoch
        BasicQLearner_Small Parking MDP_rewardList.txt: Reward List of each epoch
    '''
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Lab 3 Test 2 - Q-Learning on a Small Parking MDP, the movie, with Seed " + str(seedValue))

    # setup the mdp parameters
    numRegularSpacesPerRow = 1
    numHandicappedSpacesPerRow = 1
    busyRate = .3
    start = 4 * (numRegularSpacesPerRow + numHandicappedSpacesPerRow) - 2  # need to recompute this when we change parking lot size so the simulation starts in the right place
    mdp = createParkingMDP("basicParkingLot", numRows, numRegularSpacesPerRow, numHandicappedSpacesPerRow, busyRate, handicapBusyRate,
                           parkedReward, crashPenalty, waitingPenalty, decayBusyRate, decayReward)

    # set amount of training/testing to something small
    numSamples = 2
    numEpochs = 1
    agent = QLearningAgent("BasicQLearner", numActions, mdp.numStates, probGreedy, discountFactor, learningRate,
                           Verbosity.VERBOSE)

    trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent)


def lab3test3(seedValue):
    '''
    This is a test on a hard Parking MDP
    Your are required to run test3 to collect the reward data for creating the learning curve
    This function will create the following files for you the generate the learning curve:
        BasicQLearner_Large Parking MDP_rewardCurveData.txt: Average reward of each epoch
        BasicQLearner_Large Parking MDP_rewardList.txt: Reward List of each epoch
    '''
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 3 - Q Learning agent on a harder Parking MDP, with Seed " + str(
        seedValue))

    busyRate = .9
    mdp = createParkingMDP("busierParkingLot", numRows, numRegularSpacesPerRow, numHandicappedSpacesPerRow, busyRate, handicapBusyRate,
                           parkedReward, crashPenalty, waitingPenalty, decayBusyRate, decayReward)

    agent = QLearningAgent("BasicQLearner", numActions, mdp.numStates, probGreedy, discountFactor, learningRate,
                           Verbosity.SILENT)

    numSamples = 50
    numEpochs = 50
    trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent)
    print(agent.analyzeQfn())


############ ***Task3: Investigate the impacts of the hyperparameters of the Q-learning procedure ** ###########
# Your will run test4 and test 5 to complete task 3 by following the detailed instructions in Canvas
def lab3test4(seedValue):
    '''
    probGreedy in the Q-learning will be varied in this test
    It will create the following files for you the generate the learning curve:
        {agentname}_Large Parking MDP_rewardCurveData.txt: Average reward of each epoch
        {agentname}_Large Parking MDP_rewardList.txt: Reward List of each epoch
    '''
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 4 - Comparing 3 (or more) Q learning agents, with Seed " + str(seedValue))

    busyRate = .9
    mdp = createParkingMDP("busierParkingLot", numRows, numRegularSpacesPerRow, numHandicappedSpacesPerRow, busyRate, handicapBusyRate,
                           parkedReward, crashPenalty, waitingPenalty, decayBusyRate, decayReward)

    probGreedy = .99
    agent1 = QLearningAgent("GreedierQLearner", numActions, mdp.numStates, probGreedy, discountFactor, learningRate,
                            Verbosity.SILENT)

    probGreedy = .7
    agent2 = QLearningAgent("LessGreedyQLearner", numActions, mdp.numStates, probGreedy, discountFactor, learningRate,
                            Verbosity.SILENT)

    probGreedy = .1
    agent3 = QLearningAgent("MUCHLessGreedyQLearner", numActions, mdp.numStates, probGreedy, discountFactor, learningRate,
                            Verbosity.SILENT)

    numSamples = 500
    numEpochs = 50
    agents = [agent1, agent2, agent3]
    for agent in agents:
        trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent)
        print()


def lab3test5(seedValue):
    '''
    Learning rate in the Q-learning will be varied in this test
    It will create the following files for you the generate the learning curve:
        {agentname}_Large Parking MDP_rewardCurveData.txt: Average reward of each epoch
        {agentname}_Large Parking MDP_rewardList.txt: Reward List of each epoch
    '''
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 5 - Comparing Q learning agents in 3 (or more) different learning rates, with Seed " + str(seedValue))

    busyRate = .9
    mdp = createParkingMDP("busierParkingLot", numRows, numRegularSpacesPerRow, numHandicappedSpacesPerRow, busyRate, handicapBusyRate,
                           parkedReward, crashPenalty, waitingPenalty, decayBusyRate, decayReward)

    learningRate = 1.0
    agent1 = QLearningAgent("HighLRQLearner", numActions, mdp.numStates, probGreedy, discountFactor, learningRate,
                            Verbosity.SILENT)

    learningRate = 0.1
    agent2 = QLearningAgent("MiddleLRQLearner", numActions, mdp.numStates, probGreedy, discountFactor, learningRate,
                            Verbosity.SILENT)

    learningRate = 0.01
    agent3 = QLearningAgent("LowLRQLearner", numActions, mdp.numStates, probGreedy, discountFactor, learningRate,
                            Verbosity.SILENT)

    numSamples = 500
    numEpochs = 50
    agents = [agent1, agent2, agent3]
    for agent in agents:
        trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent)
        print()


############ ***Task4: Test the Q-learning procedure on different MDPs** ###########
def lab3test6(seedValue):
    '''
    Parking MDP with different busy rates will be created
    It will create the following files for you the generate the learning curve:
        BasicQLearner_Large Parking MDP{number}_rewardCurveData.txt: Average reward of each epoch
        BasicQLearner_Large Parking MDP{number}_rewardList.txt: Reward List of each epoch
    '''
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 6 - Comparing 3 (or more) MDPs, with Seed " + str(seedValue))

    busyRate = .99
    mdp1 = createParkingMDP("VERYbusyParkingLot", numRows, numRegularSpacesPerRow, numHandicappedSpacesPerRow, busyRate, handicapBusyRate,
                            parkedReward, crashPenalty, waitingPenalty, decayBusyRate, decayReward)
    busyRate = .5
    mdp2 = createParkingMDP("halfBusyParkingLot", numRows, numRegularSpacesPerRow, numHandicappedSpacesPerRow, busyRate, handicapBusyRate,
                            parkedReward, crashPenalty, waitingPenalty, decayBusyRate, decayReward)

    busyRate = .2
    mdp3 = createParkingMDP("mostlyEmptyParkingLot", numRows, numRegularSpacesPerRow, numHandicappedSpacesPerRow, busyRate, handicapBusyRate,
                            parkedReward, crashPenalty, waitingPenalty, decayBusyRate, decayReward)

    numSamples = 500
    numEpochs = 50
    mdps = [mdp1, mdp2, mdp3]
    for i in range(len(mdps)):
        agent = QLearningAgent("BasicQLearner", numActions, mdps[i].numStates, probGreedy, discountFactor, learningRate,
                               Verbosity.SILENT)
        mdpName = "Large Parking MDP" + str(i + 1)
        trainingAndTestLoop(numEpochs, numSamples, mdps[i], start, agent)
        print()


def lab3test7(seedValue):
    '''
    Parking MDP with different busy rates will be created
    For each MDP, Q-learning with three settings of probGreedy will be testied
    It will create the following files for you the generate the learning curve:
        {agentname}_Large Parking MDP{number}_rewardCurveData.txt: Average reward of each epoch
        {agentname}_Large Parking MDP{number}_rewardList.txt: Reward List of each epoch
    '''
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 7 - Comprehensively evaluate Q-learning parameters, with Seed " + str(seedValue))

    # make the same 3 MDPs as in the last test
    busyRate = .99
    mdp1 = createParkingMDP("VERYbusyParkingLot", numRows, numRegularSpacesPerRow, numHandicappedSpacesPerRow, busyRate,
                            handicapBusyRate,
                            parkedReward, crashPenalty, waitingPenalty, decayBusyRate, decayReward)
    busyRate = .5
    mdp2 = createParkingMDP("halfBusyParkingLot", numRows, numRegularSpacesPerRow, numHandicappedSpacesPerRow, busyRate,
                            handicapBusyRate,
                            parkedReward, crashPenalty, waitingPenalty, decayBusyRate, decayReward)

    busyRate = .2
    mdp3 = createParkingMDP("mostlyEmptyParkingLot", numRows, numRegularSpacesPerRow, numHandicappedSpacesPerRow,
                            busyRate, handicapBusyRate,
                            parkedReward, crashPenalty, waitingPenalty, decayBusyRate, decayReward)

    numSamples = 500
    numEpochs = 50
    mdps = [mdp1, mdp2, mdp3]
    for i in range(len(mdps)):
        mdpName = "Large Parking MDP" + str(i + 1)
        probGreedy = .99
        agent1 = QLearningAgent("GreedierQLearner", numActions, mdps[i].numStates, probGreedy, discountFactor, learningRate,
                                Verbosity.SILENT)
        trainingAndTestLoop(numEpochs, numSamples, mdps[i], start, agent1)
        print()

        probGreedy = .7
        agent2 = QLearningAgent("LessGreedyQLearner", numActions, mdps[i].numStates, probGreedy, discountFactor, learningRate,
                                Verbosity.SILENT)
        trainingAndTestLoop(numEpochs, numSamples, mdps[i], start, agent2)
        print()

        probGreedy = .1
        agent3 = QLearningAgent("MUCHLessGreedyQLearner", numActions, mdps[i].numStates, probGreedy, discountFactor, learningRate,
                                Verbosity.SILENT)
        trainingAndTestLoop(numEpochs, numSamples, mdps[i], start, agent3)
        print()
