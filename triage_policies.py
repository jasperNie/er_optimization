from classes import ERSimulation


def mts_policy(patients):
    """Modified Triage Score (MTS) policy based on severity and wait time"""
    if not patients:
        return None
    
    scores = []
    for patient in patients:
        # MTS uses severity as primary factor, wait time as secondary
        severity_weight = 3.0
        wait_weight = 1.0
        
        score = patient.severity * severity_weight + patient.wait_time * wait_weight
        scores.append((score, patient))
    
    # Return patient with highest MTS score
    return max(scores, key=lambda x: x[0])[1]


def esi_policy(patients):
    """Emergency Severity Index (ESI) policy focusing on urgency levels"""
    if not patients:
        return None
    
    scores = []
    for patient in patients:
        # ESI prioritizes highest severity (most urgent), then considers deterioration
        severity_weight = 4.0
        deterioration_weight = 1.5
        wait_weight = 0.5
        
        score = (patient.severity * severity_weight + 
                patient.deterioration_rate * deterioration_weight + 
                patient.wait_time * wait_weight)
        scores.append((score, patient))
    
    # Return patient with highest ESI score
    return max(scores, key=lambda x: x[0])[1]


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