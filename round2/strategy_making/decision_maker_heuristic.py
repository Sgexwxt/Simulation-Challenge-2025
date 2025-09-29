from port_simulation.entity.agv import AGV
from math import exp

class DecisionMaker:
    """
    Heuristic decision maker with downstream-urgency-aware berth allocation.
    Safe fallbacks are used whenever certain attributes are missing in the sim env,
    so this file should be drop-in compatible with your current setup.
    """

    # Tunables (you can tweak to taste)
    MAX_BERTH_CANDIDATE_VESSELS = 8          # was 3
    W_DIST_FUTURE   = 0.25
    W_SPT           = 0.60
    W_WAITED        = -0.10                  # negative -> longer wait gets priority
    W_BUSY          = 0.05
    W_URGENCY       = -0.20                  # negative -> more urgent gets priority

    # Urgency override thresholds
    DOWNSTREAM_URGENT_TH  = 0.55
    DOWNSTREAM_OVERRIDE_TH= 0.70

    def __init__(self, wsc_port=None):
        self.wsc_port = wsc_port

    # -----------------------------
    # 1) Berth allocation (updated)
    # -----------------------------
    def customized_allocated_berth(self, waiting_vessel_list):
        idle_berths = list(self.wsc_port.berth_being_idle.completed_list) if self.wsc_port else []
        if not idle_berths or not waiting_vessel_list:
            return (None, None)

        # Expand vessel candidates
        candidates = waiting_vessel_list[:self.MAX_BERTH_CANDIDATE_VESSELS]

        # Compute urgency for each candidate (safe estimation)
        vessel_urgencies = {v.id: self._estimate_downstream_urgency(v) for v in candidates}

        # If there exists "extremely urgent" vessels, soft override: restrict candidates
        max_urg = max(vessel_urgencies.values()) if vessel_urgencies else 0.0
        if max_urg >= self.DOWNSTREAM_OVERRIDE_TH:
            urgent_ids = {vid for vid, u in vessel_urgencies.items() if u >= self.DOWNSTREAM_OVERRIDE_TH}
            candidates = [v for v in candidates if v.id in urgent_ids]

        best_pair, best_score = (None, None), float('inf')
        for vessel in candidates:
            # SPT (size per required QC)
            try:
                total_boxes = sum(getattr(vessel, "discharging_containers_information", {}).values())
            except Exception:
                total_boxes = 0
            req_qc = max(1, int(getattr(vessel, "required_qc_count", 1)))
            spt = total_boxes / req_qc

            # waited seconds
            arrival_time = getattr(vessel, "arrival_time", None)
            now = getattr(self.wsc_port, "clock_time", None)
            waited = (now - arrival_time).total_seconds() if arrival_time and now else 0.0

            urgency = vessel_urgencies.get(vessel.id, 0.0)

            for berth in idle_berths:
                # future loading distance (Champion's original core)
                berth_cp = berth.equipped_qcs[1].cp if getattr(berth, "equipped_qcs", None) and len(berth.equipped_qcs) > 1 else \
                           berth.equipped_qcs[0].cp if getattr(berth, "equipped_qcs", None) else getattr(berth, "cp", None)
                dist_future = 0.0
                if berth_cp is not None and getattr(self.wsc_port, "yard_blocks", None):
                    for block in self.wsc_port.yard_blocks:
                        for c in getattr(block, "stacked_containers", []):
                            # containers planned to be loaded ON this vessel after discharge
                            if getattr(c, "loading_vessel_id", None) == vessel.id:
                                dist_future += AGV.calculate_distance(block.cp, berth_cp)

                busy_ratio = 0.0
                try:
                    eq = getattr(berth, "equipped_qcs", [])
                    busy_ratio = sum(1 for qc in eq if qc not in self.wsc_port.qc_being_idle.completed_list) / max(1, len(eq))
                except Exception:
                    busy_ratio = 0.0

                score = (
                    self.W_SPT * spt +
                    self.W_DIST_FUTURE * dist_future +
                    self.W_WAITED * waited +
                    self.W_BUSY * busy_ratio +
                    self.W_URGENCY * urgency
                )

                if score < best_score:
                    best_score, best_pair = score, (berth, vessel)

        return best_pair

    # -----------------------------
    # 2) AGV allocation (unchanged)
    # -----------------------------
    def customized_allocated_agvs(self, container):
        idle_agvs = list(self.wsc_port.agv_being_idle.completed_list) if self.wsc_port else []
        if not idle_agvs:
            return None

        best_agv, best_score = None, float('inf')
        for agv in idle_agvs:
            d1 = AGV.calculate_distance(agv.current_location, container.current_location)  # to container

            # naive predicted block -> nearest
            predicted_block = min(
                self.wsc_port.yard_blocks,
                key=lambda b: AGV.calculate_distance(container.current_location, b.cp)
            ) if getattr(self.wsc_port, "yard_blocks", None) else None

            d2 = AGV.calculate_distance(container.current_location, predicted_block.cp) if predicted_block else 0.0

            backhaul = 0.0
            if getattr(self.wsc_port, "qcs", None) and predicted_block is not None:
                backhaul = min(AGV.calculate_distance(predicted_block.cp, qc.cp) for qc in self.wsc_port.qcs)

            local_crowd = sum(
                1 for other in idle_agvs
                if other is not agv and AGV.calculate_distance(other.current_location, agv.current_location) <= 300
            )

            score = 0.4*d1 + 0.35*d2 + 0.2*backhaul + 0.05*local_crowd
            if score < best_score:
                best_score, best_agv = score, agv

        return best_agv

    # -----------------------------
    # 3) Yard-block allocation (unchanged - basic)
    # -----------------------------
    def customized_determine_yard_block(self, agv):
        blocks = [
            b for b in getattr(self.wsc_port, "yard_blocks", [])
            if len(b.stacked_containers) + b.reserved_slots < b.capacity
        ]
        if not blocks:
            return None

        cont = getattr(agv, "loaded_container", None)
        load_vid = getattr(cont, "loading_vessel_id", None) if cont else None

        best_b, best_score = None, float('inf')
        for b in blocks:
            group_bonus = -1.0 if load_vid and any(
                getattr(c, "loading_vessel_id", None) == load_vid for c in getattr(b, "stacked_containers", [])
            ) else 0.0
            bal = (len(getattr(b, "stacked_containers", [])) + getattr(b, "reserved_slots", 0)) / max(1, getattr(b, "capacity", 1))
            wait = getattr(b, "reserved_slots", 0) * 90
            dist = AGV.calculate_distance(getattr(b, "cp", None), getattr(agv, "current_location", None))

            score = 0.45*group_bonus + 0.25*bal + 0.15*wait + 0.15*dist
            if score < best_score:
                best_score, best_b = score, b

        return best_b

    # --------------------------------------
    # Helper: estimate downstream urgency u∈[0,1]
    # Higher u => more urgent.
    # --------------------------------------
    def _estimate_downstream_urgency(self, vessel):
        """
        Heuristic: how urgently should this discharging vessel be berthed
        to feed downstream loading vessels that are arriving/berthed soon.

        Implementation notes:
        - If your env exposes an explicit dependency graph (e.g. containers on `vessel`
          have `loading_vessel_id`s), replace the "soft guess" below accordingly.
        - This default uses two weak but general signals:
            (1) Presence of any other vessel already berthed (or arrived within 2h)
                whose ID appears as a `loading_vessel_id` on containers *already in yard*.
            (2) A time-based sigmoid against the earliest arrival among such vessels.
        - If signals absent, returns 0.0 (no urgency).
        """
        try:
            # Collect candidate downstream vessel IDs from yard-state:
            yard = getattr(self.wsc_port, "yard_blocks", None) or []
            dep_ids = set()
            for block in yard:
                for c in getattr(block, "stacked_containers", []):
                    # these are containers already waiting to be loaded onto some vessel
                    dv = getattr(c, "loading_vessel_id", None)
                    if dv is not None:
                        dep_ids.add(dv)

            # If this vessel contributes to any downstream (proxy: overlaps on IDs),
            # give it some baseline urgency.
            base = 0.2 if vessel.id in dep_ids else 0.0

            # Find the earliest downstream vessel arrival among currently relevant vessels
            # (berthed or arriving soon). We use a 2-hour window as "soon".
            port_vessels = list(getattr(self.wsc_port, "vessels", []))
            now = getattr(self.wsc_port, "clock_time", None)
            if not now or not port_vessels:
                return base

            soon_hours = 2.0
            mins = []
            for dv in port_vessels:
                if dv.id in dep_ids and dv is not vessel:
                    at = getattr(dv, "arrival_time", None)
                    if at:
                        delta_min = (at - now).total_seconds() / 60.0
                        mins.append(delta_min)

            if not mins:
                return base

            # Earliest (can be negative if already berthed)
            m = min(mins)
            # Map minutes to [0,1] via sigmoid around 0 (now). 0 min => ~0.5;
            # already-arrived (negative) pushes toward ~1; several hours away => small.
            u = 1.0 / (1.0 + exp(m / 60.0))  # 60 min scale
            return max(0.0, min(1.0, base + 0.8 * u))
        except Exception:
            return 0.0
