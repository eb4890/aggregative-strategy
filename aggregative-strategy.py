from enum import Enum

from typing import List, Dict

from random import Random

class State():
    max_x = 0
    max_y = 1

    min_x = 0
    min_y = 0

    def __init__(self, x : int, y: int):
        self.x = x
        self.y = y

    def random(rng: Random):
        #fix
        return State(0,rng.randrange( State.min_y, State.max_y))
    def __str__(self):
        return "x:{}, y:{}".format(self.x, self.y)
    def __repr__(self):
        return "x:{}, y:{}".format(self.x, self.y)
    def __hash__(self):
        return hash(self.x) ^ hash(self.y)

    def __eq__(self, other):
        return (self.x==other.x and self.y == other.y)

class Action(Enum):
    NORTH = 1
#    SOUTH = 2
#    EAST = 3
#    WEST = 4

class StateActionPair():
    def __init__(self, state: State, action: Action):
        self.state = state
        self.action = action

    def __eq__(self, other):
        return (self.state == other.state and self.action==other.action)

    def __hash__(self):
        return hash(self.state) ^ hash(self.action)

class ModelKnowledge:
    def __init__(self, initial_state: State,
                 action: Action,
                 resulting_state: State):
        self.initial_state = initial_state
        self.action = action
        self.resulting_state = resulting_state

    def __eq__(self, other):
        return (self.initial_state == other.initial_state and self.action==other.action and self.resulting_state ==other.resulting_state)

    def __hash__(self):
        return hash(self.initial_state) ^ hash(self.action) ^ hash(self.resulting_state)
class Agent:
    def __init__(self,
                 state_info_accuracy: float,
                 model_knowledge: ModelKnowledge,
                 preferred_state: State):
        self.state_info_accuracy = state_info_accuracy
        self.model_knowledge = model_knowledge
        self.state_preference = {}
        self.action_preference = {}
        if preferred_state is not None:
            self.state_preference[preferred_state] = 1.0

    def evaluate_state (self, state: State) -> int:
        if state in self.state_preference:
            return self.state_preference[state]
        else:
            return 0

    def evaluate_action (self, action: Action) -> int:
        if action in self.action_preference:
            return selfaction_preference[action]
        else:
            return 0

    def get_believed_state (self, state: State, rng: Random) -> State:
        chancePicked = rng.uniform(0.0,1.0)
        if chancePicked < self.state_info_accuracy:
            return state
        else:
            return State.random(rng)



class PredictionAggregation:
    def __init__(self):

        self.correct_predictions =1
        self.incorrect_predictions = 1
    def get_accuracy(self):
        return self.correct_predictions/(self.correct_predictions+self.incorrect_predictions)

class AggregationsForAction:
    def __init__(self):
        self.maximum_accuracy = 0.0
        self.most_likely_model =None
        self.prediction_aggregations = {}

    def add(self, knowledge, consensus_state: State):
        for model, aggregation in self.prediction_aggregations:
            if model.resulting_state == consensus_state:
                aggregation.correct_predictions +=1
            #Check if it has become the most accurate knowledge
            else:
                aggregation.incorrect_predictions +=1
        #Check if it has stopped being the most accurate knowledge
        #You need to iterate over all aggregations
        self.update_most_likely_model()

    def update_most_likely_model(self):
        for model, prediction_aggregation in prediction_aggregations:
            current_accuracy = prediction_aggregation.get_accuracy()
            if current_accuracy > self.maximum_accuracy:
                self.maximum_accuracy = current_accuracy
                self.most_likely_model = model



    def init_knowledge(self, knowledge):
        prediction = PredictionAggregation()
        self.prediction_aggregations[knowledge] = prediction
        if self.maximum_accuracy < 0.00000001:
            self.maximum_accuracy = 0.5
            self.most_likely_model = knowledge

class AggregationProcess:
    def __init__(self, agents_to_combine: List[Agent], initial_state: State, rng: Random):
        self.current_plan = []
        self.accurate_observations = {}
        self.accuracy_map = {}
        self.states_visited = 0
        self.previous_state = initial_state
        self.state_action_knowledge = {}
        for agent in agents_to_combine:
            self.accurate_observations[agent] = 0
            agent_state = agent.model_knowledge.initial_state
            agent_action = agent.model_knowledge.action
            if not (agent_state in self.state_action_knowledge):
                self.state_action_knowledge[agent_state] = {}
            if not (agent_action) in self.state_action_knowledge[agent_state]:
                new_action_aggregate = AggregationsForAction()
                new_action_aggregate.init_knowledge(agent.model_knowledge)
                self.state_action_knowledge[agent_state][agent_action] = new_action_aggregate


        self.previous_state = None
        self.previous_action = None
        self.prediction_depth = 3

    #This function informs the aggregation process of the current state and picks a new action to perform based upon the state
    def inform_state_and_pick_action(self, state: State):
        print (state)
        print (self.state_action_knowledge)
        state_beliefs = {}
        rng = Random()
        agent_state_beliefs = {}
        self.update_beliefs(state, rng, state_beliefs, agent_state_beliefs)
        print(state_beliefs)
        consensus_state =  self.get_consensus_state(state_beliefs)
        if self.states_visited != 0:
            self.update_prediction(consensus_state)
        if len(self.current_plan) == 0:
            self.current_plan = self.make_plan(consensus_state, 3)

        self.update_observation_accuracy(agent_state_beliefs, consensus_state)

        #Prepare for the next action
        self.previous_state = consensus_state
        current_plan_step = self.current_plan.pop(0)
        next_action = None
        if consensus_state != current_plan_step.state:
            self.make_plan(consensus_state, 3)
            current_plan_step = self.current_plan.pop(0)
            next_action = current_plan_step.action
        else:
            next_action = current_plan_step.action
        self.previous_action = next_action
        return next_action

    def update_observation_accuracy(self, agent_state_beliefs, consensus_state):
        for agent, believed_state in agent_state_beliefs.items():
            if believed_state == consensus_state:
                self.accuracy_map[agent] = self.accuracy_map.get(agent, 0) + 1

    def update_predictions(self, consensus_state):
        aggregate_for_actions =  self.state_action_knowledge[self.previous_state][self.previous_action]
        aggregate_for_actions.add(consensus_state)


    def get_consensus_state(self, state_beliefs):
        consensus_state = None
        max_believers = 0
        for state, believers in state_beliefs.items():
            if max_believers <= believers:
                max_believers = believers
                consensus_state = state
        return consensus_state

    def update_beliefs(self, state, rng, state_beliefs, agent_beliefs):
        for agent in self.accurate_observations.keys():
            believed_state = agent.get_believed_state(state, rng)
            state_beliefs[believed_state] = state_beliefs.get(believed_state, 0.0) + self.accurate_observations[agent]/(self.states_visited+1.0)
            agent_beliefs[agent] = believed_state

    def evaluate_paths (self, current_state, depth):
        allPaths = []
        if (depth == 0):
            allPaths.append(PathEvaluation())
        else:
            next_models = []
            for action in list(Action):
                print("{}{}",current_state, action, hash(current_state))
                print (self.state_action_knowledge)
                for states in self.state_action_knowledge.keys():
                    print("hash of", states, hash(states))
                print(self.state_action_knowledge[current_state])
                next_models.append( self.state_action_knowledge[current_state][action].most_likely_model)

            for models in next_models:
                sub_paths = self.evaluate_paths(models.resulting_state, depth -1)
                for path in sub_paths:
                    path.evaluateNextStateAndAction(self.accurate_observations.keys(), current_state, models.action)
                allPaths.extend(sub_paths)
        return allPaths

    def make_plan(self, current_state, depth):
        paths = []
        max_preference = 0
        current_best_path = []
        paths = self.evaluate_paths(current_state, depth)
        for path in paths:
            aggregate_preference = path.combine_preferences()
            if aggregate_preference > max_preference:
                current_best_path = path.path
                max_preference = aggregate_preference
        return current_best_path


class PathEvaluation:

    def __init__(self):
        self.evaluation =  {}
        self.path = []

    def evaluateNextStateAndAction(self,
                                   agents: List[Agent],
                                   state: State,
                                   action: Action):
        #Evaluates the current state and action for each agent
        for agent in agents:
            self.evaluation[agent] = self.evaluation.get(agent, 0) + agent.evaluate_state(state) + agent.evaluate_action(action)
        #Add the StateAction to the recorded path
        self.path.append(StateActionPair(state,action))

    def combine_preferences(self):
        # This is where you should fill in the implementation
        return 1



class Simulation:
    def __init__(self):
        self.agents = [Agent(0.7, ModelKnowledge(State(0,0), Action.NORTH, State(0,1)), State(0,1)),
                       Agent(0.7, ModelKnowledge(State(0,1), Action.NORTH, State(0,1)), State(0,1))]
        self.current_state = State(0,0)
        self.transition_table = {StateActionPair(State(0,0), Action.NORTH) : State(0,1)}
        self.aggregation_process = AggregationProcess(self.agents, self.current_state, Random())

    def tick(self):
        action = self.aggregation_process.inform_state_and_pick_action(self.current_state)
        next_state = self.transition_table[StateActionPair(self.current_state, action)]
        print("{} + {} -> {}", self.current_state, action, next_state)
        self.current_state = next_state


if __name__=="__main__":
    sim = Simulation()
    sim.tick()
