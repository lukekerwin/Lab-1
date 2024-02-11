from random import seed
from ParkingLotFactory import createParkingMDP
from agent.AgentBase import Verbosity
from agent.ProbPolicyAgents import OccupiedRandomAgent, OccupiedRandomNoHandicapAgent, OccupiedRandomNoHandicapLapAgent, YourAgent
from statistics import stdev

# The size of the action space, won't change for this lab
numActions = 2

# defines the base MDP parameters for the lot size (Don't change the values here, instead change them within your function)
numRows = 2
numRegularSpacesPerRow = 100
numHandicappedSpacesPerRow = 2

# set up a little data to create the agent and simulate the MDP
numSpacesPerRow = numRegularSpacesPerRow + numHandicappedSpacesPerRow  # agent needs this to identify space types (and other reasons for different agent types)
start = 4 * numSpacesPerRow - 2  # now that we have overall lot size, we can determine start

# defines the base MDP parameters for the transition function (Don't change the values here, instead change them within your function)
busyRate = .99
handicapBusyRate = .05

# defines the base MDP paramters for the reward function  (Don't change the values here, instead change them within your function)
parkedReward = 1000
crashPenalty = -10000
waitingPenalty = -1
decayBusyRate = False
decayReward = True

# A helper function we will use often
def printRewards(rewardList, agentName, mdpName):
    print("On\t{}\t, Over {} Trajectories, Agent\t{}\t achieved a Grand Total Reward:\t{:.4f}\t, for an average of\t{:.4f}\t with std\t {:.4f}\t. Rewards List:".format(
        mdpName, len(rewardList), agentName, sum(rewardList), sum(rewardList) / len(rewardList), stdev(rewardList)), end="\t")
    for i in range(len(rewardList)):
        print(rewardList[i], end="\t")
    print()

###### Pretask 0: Examine a simple Parking MDP ########
def lab2test0(seedValue):
    '''
    # Here we firstly create a simple parking MDP that only contains 2 rows of spaces by setting numRows as 2. 
    # Each row has 1 regular space and 1 handicapped space by setting
    #       numRegularSpacesPerRow and numHandicappedSpacesPerRow to 1.
    # The simple parking lot looks like the following (the agent will traverse the lot counter-clockwise):
    ##############################
    #           STORE            #
    # HANDICAPPED   HANDICAPPED  #
    # REGULAR       REGULAR      #  <- START
    ##############################

    # busyRate and HandicapBusyRate provide a nice way to vary the difficulty of the MDP
    # Here, they are is set to .3 and .05, respectively,  so cars will appear in regular spaces ~30% of the time and handicapped spaces ~5% of the time.
    # Note that this MDP will have 17 states:
    # 4 for each parking space and 1 for the "exit" state (sometimes useful to keep rewards from accruing after reaching a terminal node)
    '''

    print("----------------------- BEGIN Test 0 - Load and Print Simple Parking MDP")
    numRegularSpacesPerRow = 1
    numHandicappedSpacesPerRow = 1
    busyRate = .3

    mdp = createParkingMDP("basicParkingLot", numRows, numRegularSpacesPerRow, numHandicappedSpacesPerRow, busyRate, handicapBusyRate,
                           parkedReward, crashPenalty, waitingPenalty, decayBusyRate, decayReward)

    print(mdp) # Now you can visualize the mdp of the simple parking lot.


###### Task 1: Test OccupiedRandomAgent on different MDPs  ########
def lab2test1(seedValue):
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 1 - OR probabilistic policy, with Seed " + str(seedValue))

    # create the same basic MDP as in test0
    numRegularSpacesPerRow = 1
    numHandicappedSpacesPerRow = 1
    busyRate = .3
    mdp = createParkingMDP("basicParkingLot", numRows, numRegularSpacesPerRow, numHandicappedSpacesPerRow, busyRate, handicapBusyRate,
                           parkedReward, crashPenalty, waitingPenalty, decayBusyRate, decayReward)

    # need to do a special re-calculation of these values for test1 and test2 ONLY (others will use a larger lot, or not simulating on the MDP in the case of test0)
    numSpacesPerRow = numRegularSpacesPerRow + numHandicappedSpacesPerRow  # agent needs this to identify space types (and other reasons for different agent types)
    start = 4 * numSpacesPerRow - 2  # now that we have overall lot size, we can determine start

    parkProb = .5 # agent flips a coin to determine if it wants to park in an open space
    agent = OccupiedRandomAgent("OR-.5", numSpacesPerRow, parkProb, Verbosity.VERBOSE)

    # measurement loop
    numTrajectories = 10
    rewardList = []
    for _ in range(numTrajectories):
        reward = mdp.simulateTrajectory(agent, startState=start)
        rewardList.append(reward)
    printRewards(rewardList, agent.name, mdp.name)



def lab2test2(seedValue):
    '''
    # We raise the busy rates of the parking MDP and test performance again
    '''
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 2 - OR probabilistic policy on a harder MDP, with Seed " + str(seedValue))

    # create a busier MDP than in test1
    numRegularSpacesPerRow = 1
    numHandicappedSpacesPerRow = 1
    busyRate = .9
    mdp = createParkingMDP("busierParkingLot", numRows, numRegularSpacesPerRow, numHandicappedSpacesPerRow, busyRate, handicapBusyRate,
                           parkedReward, crashPenalty, waitingPenalty, decayBusyRate, decayReward)

    # need to do a special re-calculation of these values for test1 and test2 ONLY (others will use a larger lot, or not simulating on the MDP in the case of test0)
    numSpacesPerRow = numRegularSpacesPerRow + numHandicappedSpacesPerRow  # agent needs this to identify space types (and other reasons for different agent types)
    start = 4 * numSpacesPerRow - 2  # now that we have overall lot size, we can determine start

    parkProb = .5  # agent flips a coin to determine if it wants to park in an open space
    agent = OccupiedRandomAgent("OR-.5", numSpacesPerRow, parkProb, Verbosity.RESULTS)

    # measurement loop
    numTrajectories = 10
    rewardList = []
    for _ in range(numTrajectories):
        reward = mdp.simulateTrajectory(agent, startState=start)
        rewardList.append(reward)
    printRewards(rewardList, agent.name, mdp.name)



########### Task 2: Compare the the different Probabilistic Policies in the Hard parking #############
def lab2test3(seedValue):
    '''
    # Measure the performance of the next agent, the OccupiedRandomNoHandicapAgent
    '''
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 3 - ORNH probabilistic policy on a harder MDP, with Seed " + str(seedValue))

    # here we can just use default values
    mdp = createParkingMDP("hardParkingLot", numRows, numRegularSpacesPerRow, numHandicappedSpacesPerRow, busyRate, handicapBusyRate,
                           parkedReward, crashPenalty, waitingPenalty, decayBusyRate, decayReward)

    parkProb = .5  # agent flips a coin to determine if it wants to park in an open space
    agent = OccupiedRandomNoHandicapAgent("ORNH-.5", numSpacesPerRow, parkProb, Verbosity.RESULTS)

    # measurement loop
    numTrajectories = 10
    rewardList = []
    for _ in range(numTrajectories):
        reward = mdp.simulateTrajectory(agent, startState=start)
        rewardList.append(reward)
    printRewards(rewardList, agent.name, mdp.name)



def lab2test4(seedValue):
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 4 - ORNH-Half-Lap probabilistic policy on a harder MDP, with Seed " + str(seedValue))

    # here we can just use default values
    busyRate = .3
    mdp = createParkingMDP("hardParkingLot", numRows, numRegularSpacesPerRow, numHandicappedSpacesPerRow, busyRate, handicapBusyRate,
                           parkedReward, crashPenalty, waitingPenalty, decayBusyRate, decayReward)

    parkProb = .5  # agent flips a coin to determine if it wants to park in an open space
    lapFactor = 1 # agent looks to complete half of 1 lap before greedily parking in next available space
    agent = OccupiedRandomNoHandicapLapAgent("ORNH-.5Lap.5", numSpacesPerRow, parkProb, lapFactor, Verbosity.RESULTS)

    # measurement loop
    numTrajectories = 10
    rewardList = []
    for _ in range(numTrajectories):
        reward = mdp.simulateTrajectory(agent, startState=start)
        rewardList.append(reward)
    printRewards(rewardList, agent.name, mdp.name)



########### Task 3:  Design your Probabilistic Policy ##########
def lab2test5(seedValue):
    '''
    # 1. Firstly, complete the YourAgent in ProbPolicyAgents.py
    # 2. Secondly, run the following test code which reports the rewards of YourAgent by modifying the main to call this function
    # 3. Finally, report the rewards from your policy
    '''
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 5 - Your probabilistic policy on a harder MDP, with Seed " + str(seedValue))

    # here we can just use default values
    busyRate = 0.9
    mdp = createParkingMDP("hardParkingLot", numRows, numRegularSpacesPerRow, numHandicappedSpacesPerRow, busyRate, handicapBusyRate,
                           parkedReward, crashPenalty, waitingPenalty, decayBusyRate, decayReward)

    parkProb = .5  # agent flips a coin to determine if it wants to park in an open space
    lapFactor = .5 # agent looks to complete half of 1 lap before greedily parking in next available space
    agent = YourAgent("YourAgent", numSpacesPerRow, parkProb, lapFactor, Verbosity.SILENT)

    # measurement loop
    numTrajectories = 10
    rewardList = []
    for _ in range(numTrajectories):
        reward = mdp.simulateTrajectory(agent, startState=start)
        rewardList.append(reward)
    printRewards(rewardList, agent.name, mdp.name)

def lab2test6(seedValue):
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 6 - Several agents on several MDPs, with Seed " + str(
        seedValue))

    # again create the default MDP (these values are not changed from the ones above, but must be here to satisfy the interpreter)
    parkedReward = 1000
    crashPenalty = -10000
    waitingPenalty = -1
    decayBusyRate = False
    mdp1 = createParkingMDP("noDecayBRParkingLot", numRows, numRegularSpacesPerRow, numHandicappedSpacesPerRow, busyRate, handicapBusyRate,
                           parkedReward, crashPenalty, waitingPenalty, decayBusyRate, decayReward)

    # create a second MDP by inverting the rewards for parking and crashing
    parkedReward = -10000
    crashPenalty = 1000
    waitingPenalty = -1
    mdp2 = createParkingMDP("invertedParkingLot", numRows, numRegularSpacesPerRow, numHandicappedSpacesPerRow, busyRate, handicapBusyRate,
                            parkedReward, crashPenalty, waitingPenalty, decayBusyRate, decayReward)

    # create a third MDP by changing the busyRate decay behavior (based on distance from the store)
    decayBusyRate = True
    mdp3 = createParkingMDP("decayBRParkingLot", numRows, numRegularSpacesPerRow, numHandicappedSpacesPerRow, busyRate, handicapBusyRate,
                            parkedReward, crashPenalty, waitingPenalty, decayBusyRate, decayReward)

    # Now we make a few agents
    parkProb = .5  # agent flips a coin to determine if it wants to park in an open space
    lapFactor = .5
    agent1 = YourAgent("YourAgentPP.5-LF.5", numSpacesPerRow, parkProb, lapFactor, Verbosity.SILENT)

    parkProb = .9 #now agent prefers to park given the opportunity
    lapFactor = .5
    agent2 = OccupiedRandomNoHandicapLapAgent("ORNH.9-LF.5", numSpacesPerRow, parkProb, lapFactor, Verbosity.SILENT)

    parkProb = .1 #now the agent prefers to NOT park, given the opportunity
    lapFactor = 1
    agent3 = OccupiedRandomNoHandicapLapAgent("ORNH.1-LF1", numSpacesPerRow, parkProb, lapFactor, Verbosity.SILENT)

    # populate these lists with whatever agents you have created and want to test
    mdps = [mdp1, mdp2, mdp3]
    agents = [agent1, agent2, agent3]

    # measurement loop
    numTrajectories = 10
    for i in range(len(mdps)):
        for agent in agents:
            rewardList = []
            for _ in range(numTrajectories):
                reward = mdps[i].simulateTrajectory(agent, startState=start)
                rewardList.append(reward)
            printRewards(rewardList, agent.name, mdps[i].name)

def lab2test7(seedValue):
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 7 - Custom Multi-Agent Testing, with Seed " + str(
        seedValue))
    
    # MDP Values
    parkedRewards = [1000]
    crashPenalties = [-10000]
    waitingPenalties = [-1]
    decayBusyRates = [False, True]
    busyRates = [0.1, 0.5, 0.9]


    # Agent Values
    parkProbs = [0.1, 0.5, 0.9]
    lapFactors = [0.1, 0.5, 0.9]
    
    # Create MDPs
    mdps = []
    for parkedReward in parkedRewards:
        for crashPenalty in crashPenalties:
            for waitingPenalty in waitingPenalties:
                for decayBusyRate in decayBusyRates:
                    for busyRate in busyRates:
                        mdp = createParkingMDP("MDP_DBR:"+str(decayBusyRate)+"_BR:"+str(busyRate), numRows, numRegularSpacesPerRow, numHandicappedSpacesPerRow, busyRate, handicapBusyRate,
                            parkedReward, crashPenalty, waitingPenalty, decayBusyRate, decayReward)
                        mdps.append(mdp)

    # Create Agents
    agents = []

    for parkProb in parkProbs:
        for lapFactor in lapFactors:
            a1 = YourAgent("YourAgent_PP:"+str(parkProb)+"_LF:"+str(lapFactor), numSpacesPerRow, parkProb, lapFactor, Verbosity.SILENT)
            a2 = OccupiedRandomNoHandicapLapAgent("ORNH_PP:"+str(parkProb)+"_LF:"+str(lapFactor), numSpacesPerRow, parkProb, lapFactor, Verbosity.SILENT)
            a3 = OccupiedRandomNoHandicapAgent("ORNA_PP:"+str(parkProb), numSpacesPerRow, parkProb, Verbosity.SILENT)
            agents.append(a1)
            agents.append(a2)
            agents.append(a3)
    
    # measurement loop
    import pandas as pd  
    df = pd.DataFrame(columns=['MDP', 'Agent', 'Rewards'])

    numTrajectories = 10
    for i in range(len(mdps)):
        for agent in agents:
            rewardList = []
            for _ in range(numTrajectories):
                reward = mdps[i].simulateTrajectory(agent, startState=start)
                rewardList.append(reward)
            df = df.append({'MDP': mdps[i].name, 'Agent': agent.name, 'Rewards': rewardList}, ignore_index=True)

    df['RewardSum'] = df['Rewards'].apply(lambda x: sum(x))
    df['RewardMean'] = df['Rewards'].apply(lambda x: sum(x) / len(x))
    df['RewardStdev'] = df['Rewards'].apply(lambda x: stdev(x))
    df.sort_values(by='RewardSum', ascending=False, inplace=True)
    print(df.head(20))