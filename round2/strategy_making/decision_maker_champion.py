from port_simulation.entity.agv import AGV

class DecisionMaker:
    """Champion strategy-making class for port simulation based on advanced algorithms."""

    def __init__(self, wsc_port=None):
        self.wsc_port = wsc_port
        self.next_ac_pair = {}  # Dictionary for AGV-Container pairs
        self.count_single_idle_berth = 0  # Counter for when only one berth is idle
        self.function_call_count = 0  # Counter to track the number of function calls

    def customized_allocated_berth(self, waiting_vessel_list):
        """
        Allocate a berth for the given vessel based on minimum total loading distance.
        """
        if not waiting_vessel_list:
            return (None, None)
            
        allocated_berth = None
        allocated_vessel = waiting_vessel_list[0]  # Take first vessel in queue
        
        self.function_call_count += 1
        min_total_distance = float('inf')
        current_idle_berths = self.wsc_port.berth_being_idle.completed_list if self.wsc_port else []

        # If only one berth is idle, return it directly
        if len(current_idle_berths) == 1:
            self.count_single_idle_berth += 1
            return (current_idle_berths[0], allocated_vessel)

        # Loop through each idle berth
        for berth in current_idle_berths:
            total_loading_distance = 0
            berth_cp = berth.equipped_qcs[1].cp  # Assume the berth has 3 QCs, pick the middle one

            # Iterate over all yard blocks
            for yard_block in self.wsc_port.yard_blocks:
                for container in yard_block.stacked_containers:
                    # Check loading
                    if container.loading_vessel_id == allocated_vessel.id:
                        distance = AGV.calculate_distance(yard_block.cp, berth_cp)
                        total_loading_distance += distance

            if total_loading_distance < min_total_distance:
                min_total_distance = total_loading_distance
                allocated_berth = berth

        if allocated_berth is None and current_idle_berths:
            allocated_berth = current_idle_berths[0]

        # Reset counter every 30 calls
        if self.function_call_count % 30 == 0:
            self.count_single_idle_berth = 0

        return (allocated_berth, allocated_vessel)

    def customized_allocated_agvs(self, container):
        """
        Allocate an AGV for the given container using advanced scoring algorithm.
        """
        allocated_agv = None
        current_idle_agvs = self.wsc_port.agv_being_idle.completed_list if self.wsc_port else []
        
        # Return None if no idle AGV
        if len(current_idle_agvs) == 0:
            return None

        # Define State Space for DQL-like approach
        state = [
            container.current_location.x_coordinate,  # Container X-coordinate
            container.current_location.y_coordinate,  # Container Y-coordinate
            len(current_idle_agvs)  # Number of available AGVs
        ]

        # Placeholder for action-value storage
        min_score = float('inf')
        agv_scores = {}

        distance_min = 0  # Minimum possible distance
        distance_max = 2033  # Maximum distance in the port

        # Iterate through idle AGVs to compute action values (Q-values)
        for agv in current_idle_agvs:
            # Calculate distance from AGV to container
            distance_score = AGV.calculate_distance(agv.current_location, container.current_location)
            distance_normalized = (distance_score - distance_min) / (distance_max - distance_min)

            # Compute resource efficiency around AGV (nearby QCs, YardBlocks)
            nearby_resources_score = self._calculate_nearby_resources_score(agv)

            # Combine distance and resource scores for total action score
            total_score = distance_normalized * 0.9 + nearby_resources_score * 0.1

            # Store scores for Q-value update
            agv_scores[agv] = total_score

            # Select AGV with the lowest score (best Q-value)
            if total_score < min_score:
                min_score = total_score
                allocated_agv = agv

        return allocated_agv

    def _calculate_nearby_resources_score(self, agv):
        """
        Calculate the nearby resources score for an AGV.
        """
        # Parameters for detection radius and max resources
        detection_radius = 350.0
        min_resources = 0  # Minimum resource score
        max_resources = 20.0

        # Calculate nearby QCs, YardBlocks, and AGVs
        nearby_qcs = sum(1 for qc in self.wsc_port.qcs 
                        if AGV.calculate_distance(qc.cp, agv.current_location) <= detection_radius 
                        and qc not in self.wsc_port.qc_being_idle.completed_list)

        nearby_yard_blocks = sum(1 for block in self.wsc_port.yard_blocks 
                               if AGV.calculate_distance(block.cp, agv.current_location) <= detection_radius 
                               and self._has_active_tasks(block))

        nearby_agvs = sum(1 for other_agv in self.wsc_port.agv_being_idle.completed_list 
                         if AGV.calculate_distance(other_agv.current_location, agv.current_location) <= detection_radius)

        # Compute total resources score with weighting factors
        total_resources = 5 * nearby_qcs + 5 * nearby_yard_blocks - 10 * nearby_agvs

        # Normalize resource score
        resources_normalized = (total_resources - min_resources) / (max_resources - min_resources)
        return max(0, min(1, resources_normalized))  # Clamp between 0 and 1

    def _has_active_tasks(self, block):
        """
        Helper function to check active tasks for YardBlock.
        """
        return block.reserved_slots > 0

    def customized_determine_yard_block(self, agv):
        """
        Determine the yard block to allocate for the given AGV.
        """
        return self._determine_yard_block(agv)

    def _determine_yard_block(self, agv):
        """
        Advanced yard block determination considering distance and wait time.
        """
        yard_blocks = self.wsc_port.yard_blocks if self.wsc_port else []

        available_blocks = [block for block in yard_blocks 
                          if block.capacity > block.reserved_slots + len(block.stacked_containers)]

        if len(available_blocks) == 0:
            print("No available yard blocks")
            return None

        # Sort blocks by distance and take top 5 candidates
        sorted_blocks = sorted(available_blocks, 
                             key=lambda block: AGV.calculate_distance(block.cp, agv.current_location))
        candidate_blocks = sorted_blocks[:5]

        min_time_cost = float('inf')
        best_block = None

        for block in candidate_blocks:
            wait_time_for_yc = self._calculate_wait_time_for_yc(block)
            transport_time = AGV.calculate_distance(block.cp, agv.current_location) / agv.speed
            total_time_cost = wait_time_for_yc + transport_time
            
            if total_time_cost < min_time_cost:
                min_time_cost = total_time_cost
                best_block = block

        return best_block

    def _calculate_wait_time_for_yc(self, block):
        """
        Calculate estimated wait time for YC at a yard block.
        """
        reserved_slots = block.reserved_slots
        processing_time_per_container = 90  # seconds
        total_workload = reserved_slots
        estimated_wait_time = total_workload * processing_time_per_container

        return estimated_wait_time

    def determine_qc(self, container):
        """
        Determine the QC (Quay Crane) for the given container.
        """
        berth = next(
            (b for b in self.wsc_port.berths if b.berthed_vessel and
             b.berthed_vessel.id == container.loading_vessel_id and
             b.berthed_vessel.week > container.week),
            None
        )

        if berth is None:
            return None

        qc = berth.equipped_qcs[berth.current_work_qc]
        berth.current_work_qc += 1
        if berth.current_work_qc == 3:
            berth.current_work_qc = 0

        return qc