#!/usr/bin/env python3
"""
Advanced Patient Arrival Patterns
Provides various realistic arrival patterns for ER simulation testing
"""

import random
import math
from classes import Patient

def generate_standard_arrivals(total_time, arrival_prob, seed):
    """Original sinusoidal pattern with daily cycle and bursts"""
    random.seed(seed)
    arrivals = []
    patient_id = 0
    timesteps_per_day = 96
    
    for t in range(total_time):
        base_prob = arrival_prob
        time_factor = 0.15 * (1 + math.sin(2 * math.pi * t / timesteps_per_day))
        burst = 0.15 if random.random() < 0.05 else 0
        prob = min(1.0, max(0.0, base_prob + time_factor + burst))
        
        if random.random() < prob:
            patient = create_patient(patient_id)
            patient_id += 1
            arrivals.append(patient)
        else:
            arrivals.append(None)
    
    return arrivals

def generate_peak_hours_arrivals(total_time, arrival_prob, seed):
    """Pattern with distinct morning and evening peaks like real EDs"""
    random.seed(seed)
    arrivals = []
    patient_id = 0
    timesteps_per_day = 96  # 24 hours in 15-min intervals
    
    for t in range(total_time):
        hour_of_day = (t % timesteps_per_day) * 15 / 60  # Convert to hour 0-24
        
        base_prob = arrival_prob
        
        # Morning peak (7am-10am): 40% increase
        if 7 <= hour_of_day < 10:
            peak_factor = 0.4
        # Lunch rush (11am-2pm): 25% increase  
        elif 11 <= hour_of_day < 14:
            peak_factor = 0.25
        # Evening peak (6pm-9pm): 50% increase
        elif 18 <= hour_of_day < 21:
            peak_factor = 0.5
        # Night hours (10pm-6am): 60% decrease
        elif hour_of_day >= 22 or hour_of_day < 6:
            peak_factor = -0.6
        else:
            peak_factor = 0
            
        # Add random bursts during peak hours
        burst = 0
        if peak_factor > 0 and random.random() < 0.08:  # Higher burst chance during peaks
            burst = 0.2
        elif random.random() < 0.03:  # Regular burst chance
            burst = 0.1
            
        prob = min(1.0, max(0.05, base_prob + base_prob * peak_factor + burst))
        
        if random.random() < prob:
            patient = create_patient(patient_id)
            patient_id += 1
            arrivals.append(patient)
        else:
            arrivals.append(None)
    
    return arrivals

def generate_weekend_arrivals(total_time, arrival_prob, seed):
    """Weekend pattern with late night peaks and different severity distribution"""
    random.seed(seed)
    arrivals = []
    patient_id = 0
    timesteps_per_day = 96
    
    for t in range(total_time):
        hour_of_day = (t % timesteps_per_day) * 15 / 60
        
        base_prob = arrival_prob
        
        # Weekend pattern: higher late night activity
        if 20 <= hour_of_day or hour_of_day < 4:  # 8pm-4am party hours
            peak_factor = 0.6
        elif 10 <= hour_of_day < 16:  # Late morning to afternoon
            peak_factor = 0.3
        else:  # Early morning, quiet
            peak_factor = -0.4
            
        # More frequent but smaller bursts on weekends
        burst = 0
        if random.random() < 0.07:
            burst = 0.15
            
        prob = min(1.0, max(0.05, base_prob + base_prob * peak_factor + burst))
        
        if random.random() < prob:
            # Weekend patients: more trauma, alcohol-related
            patient = create_patient(patient_id, weekend_bias=True)
            patient_id += 1
            arrivals.append(patient)
        else:
            arrivals.append(None)
    
    return arrivals

def generate_disaster_arrivals(total_time, arrival_prob, seed):
    """Disaster scenario with massive influx at specific time"""
    random.seed(seed)
    arrivals = []
    patient_id = 0
    timesteps_per_day = 96
    
    # Disaster occurs at random time between 8am and 8pm
    disaster_time = random.randint(32, 80)  # 8am-8pm in timesteps
    disaster_duration = 12  # 3 hours of elevated arrivals
    
    for t in range(total_time):
        base_prob = arrival_prob
        
        # Normal daily pattern
        time_factor = 0.1 * (1 + math.sin(2 * math.pi * t / timesteps_per_day))
        
        # Disaster surge
        if disaster_time <= t < disaster_time + disaster_duration:
            # Massive spike during disaster
            surge_intensity = 3.0 if t < disaster_time + 4 else 2.0  # Peak in first hour
            disaster_factor = surge_intensity
        elif disaster_time + disaster_duration <= t < disaster_time + disaster_duration + 8:
            # Aftermath - elevated but declining
            disaster_factor = 1.0 * (1 - (t - disaster_time - disaster_duration) / 8)
        else:
            disaster_factor = 0
            
        prob = min(1.0, max(0.0, base_prob + time_factor + disaster_factor))
        
        if random.random() < prob:
            # During disaster: more severe injuries
            is_disaster_victim = disaster_time <= t < disaster_time + disaster_duration * 2
            patient = create_patient(patient_id, disaster_victim=is_disaster_victim)
            patient_id += 1
            arrivals.append(patient)
        else:
            arrivals.append(None)
    
    return arrivals

def generate_flu_season_arrivals(total_time, arrival_prob, seed):
    """Flu season pattern with increased respiratory cases and volume"""
    random.seed(seed)
    arrivals = []
    patient_id = 0
    timesteps_per_day = 96
    
    for t in range(total_time):
        base_prob = arrival_prob
        
        # Daily pattern with morning peak (people realize they're sick)
        hour_of_day = (t % timesteps_per_day) * 15 / 60
        if 8 <= hour_of_day < 12:  # Morning peak
            time_factor = 0.3
        elif 14 <= hour_of_day < 18:  # Afternoon
            time_factor = 0.2
        elif hour_of_day >= 20 or hour_of_day < 6:  # Night (less flu visits)
            time_factor = -0.3
        else:
            time_factor = 0
            
        # Flu season: 40% overall increase in volume
        flu_factor = 0.4
        
        # Occasional mini-surges (families getting sick together)
        surge = 0
        if random.random() < 0.06:  # Family clusters
            surge = 0.25
            
        prob = min(1.0, max(0.05, base_prob + base_prob * flu_factor + base_prob * time_factor + surge))
        
        if random.random() < prob:
            patient = create_patient(patient_id, flu_season=True)
            patient_id += 1
            arrivals.append(patient)
        else:
            arrivals.append(None)
    
    return arrivals

def generate_steady_state_arrivals(total_time, arrival_prob, seed):
    """Steady state with minimal variation for baseline testing"""
    random.seed(seed)
    arrivals = []
    patient_id = 0
    
    for t in range(total_time):
        # Very minimal variation - just small random fluctuations
        variation = random.uniform(-0.05, 0.05)
        prob = min(1.0, max(0.0, arrival_prob + variation))
        
        if random.random() < prob:
            patient = create_patient(patient_id)
            patient_id += 1
            arrivals.append(patient)
        else:
            arrivals.append(None)
    
    return arrivals

def create_patient(patient_id, weekend_bias=False, disaster_victim=False, flu_season=False):
    """Create a patient with appropriate characteristics based on scenario"""
    
    if disaster_victim:
        # Disaster victims: higher severity, more trauma
        severity = random.choices([1,2,3,4,5], weights=[0.05,0.1,0.2,0.3,0.35])[0]
        presenting_choices = ['major_trauma', 'burns', 'crush_injury', 'chest_pain', 'head_injury']
    elif weekend_bias:
        # Weekend: more trauma and alcohol-related
        severity = random.choices([1,2,3,4,5], weights=[0.08,0.12,0.25,0.3,0.25])[0]  
        presenting_choices = ['trauma', 'alcohol_poisoning', 'assault', 'motor_vehicle_accident', 'sports_injury']
    elif flu_season:
        # Flu season: more respiratory, generally less severe
        severity = random.choices([1,2,3,4,5], weights=[0.15,0.25,0.35,0.2,0.05])[0]
        presenting_choices = ['shortness_of_breath', 'fever', 'cough', 'flu_like_symptoms', 'respiratory_distress']
    else:
        # Standard case mix
        severity = random.choices([1,2,3,4,5], weights=[0.1,0.15,0.25,0.25,0.25])[0]
        presenting_choices = ['chest_pain', 'abdominal_pain', 'shortness_of_breath', 'minor_injury', 'headache']
    
    # Deterioration correlates with severity
    det_base = 0.1 + 0.15 * (severity-1)
    deterioration = round(random.uniform(det_base, min(0.6, det_base+0.15)), 2)
    
    # Treatment time varies by severity and scenario
    treat_base = 5 + 2 * severity
    if disaster_victim:
        treat_base += 5  # Disaster victims take longer
    elif flu_season and severity <= 2:
        treat_base = max(3, treat_base - 2)  # Simple flu cases are quicker
    
    treatment_time = random.randint(treat_base, treat_base+5)
    
    # Create patient
    patient = Patient(patient_id, severity, deterioration, treatment_time)
    
    # Add scenario-specific attributes
    patient.presenting = random.choice(presenting_choices)
    patient.vitals = generate_vitals(severity, disaster_victim)
    patient.expected_resources = random.choice([0, 1, 2, 3])
    
    return patient

def generate_vitals(severity, disaster_victim=False):
    """Generate realistic vital signs based on severity"""
    if disaster_victim and severity >= 4:
        # Critical disaster victims
        return {
            'hr': random.randint(110, 150),
            'rr': random.randint(20, 35),
            'spo2': random.randint(85, 95),
            'bp_sys': random.randint(70, 110)
        }
    elif severity >= 4:
        # Critical patients
        return {
            'hr': random.randint(100, 140),
            'rr': random.randint(18, 32),
            'spo2': random.randint(88, 96),
            'bp_sys': random.randint(80, 120)
        }
    elif severity >= 3:
        # Moderate patients
        return {
            'hr': random.randint(80, 120),
            'rr': random.randint(14, 24),
            'spo2': random.randint(92, 98),
            'bp_sys': random.randint(90, 150)
        }
    else:
        # Stable patients
        return {
            'hr': random.randint(60, 100),
            'rr': random.randint(12, 20),
            'spo2': random.randint(96, 100),
            'bp_sys': random.randint(100, 140)
        }

# Pattern registry for easy access
ARRIVAL_PATTERNS = {
    'standard': generate_standard_arrivals,
    'peak_hours': generate_peak_hours_arrivals, 
    'weekend': generate_weekend_arrivals,
    'disaster': generate_disaster_arrivals,
    'flu_season': generate_flu_season_arrivals,
    'steady_state': generate_steady_state_arrivals
}

def get_pattern_description(pattern_name):
    """Get human-readable description of arrival pattern"""
    descriptions = {
        'standard': 'Original sinusoidal daily cycle with random bursts',
        'peak_hours': 'Realistic weekday with morning/evening peaks and quiet nights',
        'weekend': 'Weekend pattern with late-night activity and trauma bias',
        'disaster': 'Mass casualty incident with surge followed by aftermath',
        'flu_season': 'Seasonal epidemic with increased respiratory cases',
        'steady_state': 'Minimal variation baseline for controlled testing'
    }
    return descriptions.get(pattern_name, 'Unknown pattern')

def log_pattern_info(pattern_name, seed, log_dir, context="Testing"):
    """Log information about the arrival pattern used"""
    import os
    os.makedirs(log_dir, exist_ok=True)
    
    with open(f"{log_dir}/arrival_pattern_info.txt", "a") as f:
        f.write(f"\n{context} - Seed {seed}:\n")
        f.write(f"   Pattern: {pattern_name}\n")
        f.write(f"   Description: {get_pattern_description(pattern_name)}\n")
        f.write(f"   Generated at: {__import__('datetime').datetime.now()}\n")