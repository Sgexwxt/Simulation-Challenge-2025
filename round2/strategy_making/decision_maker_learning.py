class DecisionMaker:
    def __init__(self, wsc_port=None, rl=None):
        self.wsc_port = wsc_port
        self.reinforcing_learning = rl
        self.state_action_history = {}  # Used to store state and action for each vessel

    def customized_allocated_berth(self, waiting_vessel_list):
        """
        Allocate a berth for the given vessel.
        """
        if self.reinforcing_learning != None:
            # Get current berth status
            action_state_berth = [1 if b in self.wsc_port.berth_being_idle.completed_list else 0 for b in self.wsc_port.berths]
            
            max_vessels_in_decision_pool = 3
            num_vessels_to_actually_consider = min(len(waiting_vessel_list), max_vessels_in_decision_pool)
            action_state_vessel = [1 if i < num_vessels_to_actually_consider else 0 for i in range(max_vessels_in_decision_pool)]
            
            #print(f"state of berths:{state}")
            if sum(action_state_berth) == 0 or sum(action_state_vessel) == 0:
                #print("No available berths. Returning None.")
                return (None, None)
    
            env_state_berth =[]
            env_state_vessel =[]
    
            # Use reinforcement learning model to select actions
            action_berth, prob_berth = self.reinforcing_learning.select_action(action_state_berth, env_state_berth, agent_idx=0)
            self.reinforcing_learning.store_transition(0, action_state_berth+env_state_berth, action_berth, 0, False, prob_berth)
            
            action_vessel, prob_vessel = self.reinforcing_learning.select_action(action_state_vessel, env_state_vessel, agent_idx=3)
            self.reinforcing_learning.store_transition(3, action_state_vessel+env_state_vessel, action_vessel, 0, False, prob_berth)
            
            allocated_berth = self.wsc_port.berths[action_berth]
            allocated_vessel = waiting_vessel_list[action_vessel]
            
            # Record state and action
            if allocated_vessel not in self.state_action_history:
                self.state_action_history[allocated_vessel]  = {}
            self.state_action_history[allocated_vessel]["berth"] = (action_state_berth+env_state_berth, action_berth, prob_berth)
            
            if allocated_berth not in self.state_action_history:
                self.state_action_history[allocated_berth]  = {}
            self.state_action_history[allocated_berth]["vessel"] = (action_state_vessel+env_state_vessel, action_vessel, prob_vessel)
            
            try:
                return (allocated_berth, allocated_vessel)
            except IndexError:
                #print(f"Invalid berth action. Returning None.[{action}]")
                return (None, None)
        else:
            return (None, None)

    def customized_allocated_agvs(self, container):
        """
        Allocate an AGV for the given vessel.
        """
        # Get current AGV status
        if self.reinforcing_learning != None:
            action_state = [1 if a in self.wsc_port.agv_being_idle.completed_list else 0 for a in self.wsc_port.agvs]
            #print(f"state of agvs:{action_state}")
            if sum(action_state) == 0:
                #print("No available AGVs. Returning None.")
                return None
            
            env_state =[]
            
            #extra information: Provide distance between AGV and container
            # Calculate distance between each AGV and container
            #from port_simulation.entity.agv import AGV
            #list_of_distances = [AGV.calculate_distance(container.current_location, agv.current_location) for agv in self.wsc_port.agvs]
            #print(list_of_distances )
            #env_state = list_of_distances
            
            # Use reinforcement learning model to select action
            action, prob = self.reinforcing_learning.select_action(action_state, env_state, agent_idx=1)
            self.reinforcing_learning.store_transition(1, action_state+env_state, action, 0, False, prob)
    
            # Record state and action
            if container not in self.state_action_history:
                self.state_action_history[container] = {}
            self.state_action_history[container]["agv"] = (action_state+env_state, action, prob)
    
            try:
                allocated_agv = self.wsc_port.agvs[action]
                #print(f"Allocated AGV: {allocated_agv}")
                return allocated_agv
            except IndexError:
                #print(f"Invalid AGV action. Returning None.[{action}]")
                return None
        else:
            return None

    def customized_determine_yard_block(self, agv):
        """
        Determine the yard block to allocate for the given vessel.
        """
        # Get current yard block status
        if self.reinforcing_learning != None:
            action_state = [1 if y.capacity > y.reserved_slots + len(y.stacked_containers) else 0 for y in self.wsc_port.yard_blocks]
            #print(f"state of YBs:{state}")
            if sum(action_state) == 0:
                #print("No available yard blocks. Returning None.")
                return None

            env_state =[]

            # Use reinforcement learning model to select action
            action, prob = self.reinforcing_learning.select_action(action_state, env_state, agent_idx=2)
            self.reinforcing_learning.store_transition(2, action_state+env_state, action, 0, False, prob)
    
            # Record state and action
            if agv not in self.state_action_history:
                self.state_action_history[agv] = {}
            self.state_action_history[agv]["yard_block"] = (action_state+env_state, action, prob)
    
            try:
                allocated_yard_block = self.wsc_port.yard_blocks[action]
                #print(f"Allocated yard block: [{action}]")
                return allocated_yard_block
            except IndexError:
                #print(f"Invalid yard block action. Returning None.[{action}]")
                return None
        else:
            return None

    def get_reward_and_update(self, vessel):
        """
        Calculate reward and update the RL model.
        """
        if self.reinforcing_learning != None:
            assert self.reinforcing_learning is not None, "Reinforcing learning instance is None."
    
            # Calculate reward
            #service_time = (vessel.departure_time - vessel.start_berthing_time).total_seconds() / 3600
            reward = -vessel.total_time  # Example reward
    
            # Retrieve stored states and actions
            vessel_history = self.state_action_history.get(vessel, {})
            for agent_idx, key in enumerate(["berth", "agv", "yard_block", "vessel"]):
                if key in vessel_history:
                    state, action, prob = vessel_history[key]
                    self.reinforcing_learning.store_transition(agent_idx, state, action, reward, True, prob)
    
            self.reinforcing_learning.update()
            #print(f"Updated RL policy with reward: {reward}")