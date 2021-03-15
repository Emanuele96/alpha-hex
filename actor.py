import variables
import random
class Actor:

    def __init__(self):
        self.policy = {}
        self.SAP_eligibilities = {}
        self.e_greedy = variables.e_actor_start
        self.lr = variables.lr_actor
        self.eligibility_decay = variables.eligibility_decay_actor
        self.discount = variables.discount_actor
        self.SAPs_in_episode = list(())
        self.counter = 0 
        random.seed(variables.random_seed_actor)

    def get_action(self, state, possible_actions):
      
        #Get all SAP for the given state
        if random.uniform(0,1) <= self.e_greedy:
            #Do a greedy choice
            choosen_action = random.choice(possible_actions)
        else:  
            #Retrieve the SAP with the highest value
            choosen_action = possible_actions[0]
            max_policy_value =  self.policy.setdefault((state,possible_actions[0]), 0)
            for action in possible_actions:
                policy_value = self.policy.setdefault((state,action), 0)
                if  policy_value > max_policy_value:
                    choosen_action = action
                    max_policy_value = policy_value

        return choosen_action

    def update(self, state_t, action_t, TD_error):
        #Get updated TD-error, update policy and eligibilities
        ## Set eligibility last state action pair to be 1
        self.SAP_eligibilities[(state_t, action_t)] = 1

        #Add the SAP in the list of SAPs seen in this episode
        if (state_t, action_t) not in self.SAPs_in_episode:
            self.SAPs_in_episode.append((state_t,action_t))

        for SAP in self.SAPs_in_episode:
            self.policy[SAP] =  self.policy.setdefault(SAP, 0) + self.lr * TD_error * self.SAP_eligibilities[SAP]
            self.SAP_eligibilities[SAP] = self.eligibility_decay * self.discount * self.SAP_eligibilities[SAP]
        
    def reset(self):
        self.counter = self.counter + 1
        self.SAPs_in_episode.clear()
        self.SAP_eligibilities.clear()
        self.e_decay()

    def e_decay(self):
        #Decay the e sigma factor for the e greedy strategy.
        if self.counter >= variables.episodes - variables.episodes * variables.total_greedy_percent:
            self.e_greedy = 0
        elif variables.decay_function == "decay":
            self.e_greedy = max(self.e_greedy* variables.e_decay,0)    
        elif variables.decay_function == "variable_decay":
            n_steps = variables.episodes
            decay_step = 5/n_steps
            self.e_greedy = max(pow(1 - decay_step, self.counter),0)
        elif variables.decay_function == "linear":
            self.e_greedy = max(self.e_greedy - ((variables.e_actor_start - variables.e_actor_stop)/ variables.episodes),0)