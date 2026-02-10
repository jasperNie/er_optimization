from classes import ERSimulation


def mts_policy(patient):
    """Manchester Triage System (MTS) policy - returns priority score for a single patient"""
    if patient is None:
        return 0
    
    # Map severity to MTS colors with time targets:
    # Red (5): 0 min, Orange (4): 10 min, Yellow (3): 60 min, Green (2): 120 min, Blue (1): 240 min
    base_priority = patient.severity * 1000  # Base priority from severity
    
    # Use deterioration chance as clinical instability indicator
    # High deterioration risk bumps up urgency (like MTS discriminators)
    if patient.deterioration_chance > 0.4:  # High risk
        base_priority += 500  # Significant bump up
    elif patient.deterioration_chance > 0.25:  # Moderate risk  
        base_priority += 200  # Moderate bump up
        
    # Wait time as tie-breaker (MTS considers current wait vs target)
    # Longer waits get slight priority boost
    base_priority += patient.wait_time * 10
    
    return base_priority


def esi_policy(patient):
    """Emergency Severity Index (ESI) policy - returns priority score for a single patient"""
    if patient is None:
        return 0
        
    def get_esi_level(patient):
        """Calculate ESI level (1-5) using official algorithm"""
        # Step 1: Life-saving intervention needed right now?
        # Map severity 5 + high deterioration to ESI 1 (immediate)
        if patient.severity == 5 and patient.deterioration_chance > 0.5:
            return 1
            
        # Step 2: High risk / can't safely wait?
        # High severity (4-5) or high deterioration risk → ESI 2
        if (patient.severity >= 4 or 
            patient.deterioration_chance > 0.4 or 
            patient.wait_time > 120):  # Severe distress from long wait
            return 2
            
        # Step 3: Resource count estimation
        # Map treatment_time to resource needs:
        # Short treatment (5-10 min) = 1 resource
        # Medium treatment (11-20 min) = 2+ resources  
        # Long treatment (21+ min) = 2+ resources
        if patient.treatment_time <= 10:
            if patient.severity <= 1:  # Very minor
                return 5  # 0 resources
            else:
                return 4  # 1 resource
        else:
            # 2+ resources → provisional ESI 3
            # Step 4: Danger zone vitals check
            # Use deterioration chance as proxy for unstable vitals
            if patient.deterioration_chance > 0.3:  # Danger zone
                return 2  # Uptriage to ESI 2
            return 3
    
    esi_level = get_esi_level(patient)
    # ESI 1 = highest priority, ESI 5 = lowest
    base_priority = (6 - esi_level) * 1000
    
    # Within same ESI level, prioritize by wait time
    base_priority += patient.wait_time * 10
    
    return base_priority


def compare_policies(num_nurses=3, total_time=96, arrival_prob=0.5, seed=123):
    """Compare MTS and ESI policies"""
    policies = {
        'MTS': mts_policy,
        'ESI': esi_policy
    }
    
    results = {}
    
    for name, policy in policies.items():
        sim = ERSimulation(
            num_nurses=num_nurses,
            total_time=total_time,
            arrival_prob=arrival_prob,
            triage_policy=policy,
            verbose=False,
            seed=seed
        )
        results[name] = sim.run()
    
    return results