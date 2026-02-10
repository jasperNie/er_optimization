import random
import statistics
import math

# -------------------------
# Patient & Nurse Classes
# -------------------------

class Patient:
    def __init__(self, id, severity, deterioration_chance, treatment_time):
        self.id = id
        self.severity = severity  # e.g., 1–5 (5 = most severe)
        self.deterioration_chance = deterioration_chance  # float [0,1]
        self.treatment_time = treatment_time  # minutes needed
        self.wait_time = 0

    def __repr__(self):
        return (f"Patient({self.id}, sev={self.severity}, "
                f"det={self.deterioration_chance:.2f}, "
                f"treat={self.treatment_time}, wait={self.wait_time})")


class Nurse:
    def __init__(self, id):
        self.id = id
        self.busy_until = 0  # time until nurse is free
        self.current_patient = None

    def __repr__(self):
        return f"Nurse({self.id}, busy_until={self.busy_until})"


# -------------------------
# ER Simulation Class
# -------------------------


class ERSimulation:
    def __init__(self, num_nurses=3, total_time=50, arrival_prob=0.3, triage_policy=None, verbose=False, seed=None, use_shifts=True):
        self.time = 0
        self.total_time = total_time
        self.arrival_prob = arrival_prob
        self.base_nurses = [Nurse(i) for i in range(num_nurses)]
        self.waiting_patients = []
        self.completed_patients = []
        self.next_patient_id = 0
        self.started_patients = []  # track when patients actually start treatment
        self.triage_policy = triage_policy or {'severity': 1.0, 'deterioration': 1.0, 'wait_time': 1.0}
        self.verbose = verbose
        self.seed = seed
        self.use_shifts = use_shifts
        
        # CRITICAL FIX: Use separate RNG for patient generation vs deterioration
        # This ensures identical patient populations regardless of triage policy
        if seed is not None:
            random.seed(seed)
            # Create separate RNG for deterioration that won't be affected by triage policy
            self.deterioration_rng = random.Random(seed + 999999)
        else:
            self.deterioration_rng = random.Random()
        
        # Nursing schedule: 8-hour shifts (1 timestep = 15 minutes)
        # 8 hours = 32 timesteps (8 * 60 / 15 = 32)
        # Day shift: 0-32, Evening shift: 32-64, Night shift: 64-96 (then repeats)
        self.timesteps_per_hour = 4  # 60 minutes / 15 minutes = 4 timesteps per hour
        self.timesteps_per_shift = 8 * self.timesteps_per_hour  # 32 timesteps per 8-hour shift
        self.timesteps_per_day = 24 * self.timesteps_per_hour   # 96 timesteps per 24-hour day
        
        self.shift_schedule = {
            'day': (0, 32),       # timesteps 0-31 (6 AM - 2 PM)
            'evening': (32, 64),  # timesteps 32-63 (2 PM - 10 PM)  
            'night': (64, 96)     # timesteps 64-95 (10 PM - 6 AM)
        }
        
        # Different staffing levels per shift (multiplier of base_nurses)
        self.shift_multipliers = {
            'day': 1.5,      # More nurses during day
            'evening': 1.0,   # Normal staffing
            'night': 0.7      # Fewer nurses at night
        }
        # Pre-generate patient arrivals and attributes for each timestep
        import math
        self.patient_arrivals = []  # List of (timestep, Patient) or None
        for t in range(self.total_time):
            # More realistic, time-varying and bursty arrival probability
            base_prob = self.arrival_prob
            time_factor = 0.15 * (1 + math.sin(2 * math.pi * t / self.timesteps_per_day))  # daily cycle
            burst = 0.15 if random.random() < 0.05 else 0  # occasional burst
            prob = min(1.0, max(0.0, base_prob + time_factor + burst))
            if random.random() < prob:
                # Correlate severity, deterioration, and treatment time
                severity = random.choices([1,2,3,4,5], weights=[0.1,0.15,0.25,0.25,0.25])[0]
                det_base = 0.1 + 0.15 * (severity-1)
                deterioration = round(random.uniform(det_base, min(0.6, det_base+0.15)), 2)
                treat_base = 5 + 2 * severity
                treatment_time = random.randint(treat_base, treat_base+5)
                patient = Patient(self.next_patient_id, severity, deterioration, treatment_time)
                self.next_patient_id += 1
                self.patient_arrivals.append(patient)
            else:
                self.patient_arrivals.append(None)
        if self.verbose:
            self.print_arrivals()
    
    def get_current_nurses(self):
        """Get the number of nurses available at the current time based on shift schedule"""
        if not self.use_shifts:
            return self.base_nurses
            
        # Determine current shift (24-hour cycle in 15-minute timesteps)
        timestep_in_day = self.time % self.timesteps_per_day  # 96 timesteps per day
        
        if 0 <= timestep_in_day < 32:  # timesteps 0-31 (6 AM - 2 PM)
            shift = 'day'
        elif 32 <= timestep_in_day < 64:  # timesteps 32-63 (2 PM - 10 PM)
            shift = 'evening'
        else:  # timesteps 64-95 (10 PM - 6 AM)
            shift = 'night'
            
        # Calculate number of nurses for this shift
        base_count = len(self.base_nurses)
        shift_count = max(1, int(base_count * self.shift_multipliers[shift]))
        
        # Handle shift transitions: preserve nurses who are currently treating patients
        current_nurses = []
        
        # First, carry over any nurses who are currently treating patients (continuity of care)
        if hasattr(self, 'nurses') and self.nurses:
            for existing_nurse in self.nurses:
                if existing_nurse.current_patient is not None:
                    # Nurse stays to complete current patient treatment (handoff protocol)
                    existing_nurse.id = f"{shift}_carryover_{existing_nurse.id}"
                    current_nurses.append(existing_nurse)
                    if self.verbose and timestep_in_day in [0, 32, 64]:
                        print(f"[t={self.time}] Nurse {existing_nurse.id} staying to complete treatment of Patient {existing_nurse.current_patient.id}")
        
        # Then fill remaining positions with new shift nurses
        nurses_needed = max(0, shift_count - len(current_nurses))
        for i in range(nurses_needed):
            if i < len(self.base_nurses):
                # Reuse base nurse IDs when possible
                nurse = Nurse(f"{shift}_{self.base_nurses[i].id}")
            else:
                # Create additional nurses for busier shifts
                nurse = Nurse(f"{shift}_extra_{i}")
            current_nurses.append(nurse)
                
        return current_nurses

    def print_arrivals(self, file_path="arrivals_log.txt"):
        with open(file_path, "w") as f:
            f.write("[ERSimulation] Pre-generated patients (None = no arrival):\n")
            for t, p in enumerate(self.patient_arrivals, 1):
                f.write(f"t={t}: {p}\n")
    
    def print_nurse_schedule(self, file_path="nurse_schedule_log.txt"):
        """Print the nursing schedule for the entire simulation to a file"""
        with open(file_path, "w") as f:
            f.write("[ERSimulation] Nursing Schedule (15-minute intervals):\n")
            f.write("Format: Timestep | Time | Shift | Nurses on Duty\n")
            f.write("-" * 50 + "\n")
            
            for t in range(1, self.total_time + 1):
                # Calculate time representation
                minutes_elapsed = (t - 1) * 15
                hours = (minutes_elapsed // 60) % 24
                minutes = minutes_elapsed % 60
                time_str = f"{hours:02d}:{minutes:02d}"
                
                # Determine shift and nurse count
                timestep_in_day = (t - 1) % self.timesteps_per_day
                
                # Mark shift changes BEFORE printing the timestep info
                if timestep_in_day in [0, 32, 64]:
                    f.write("    >>> SHIFT CHANGE <<<\n")
                
                if 0 <= timestep_in_day < 32:
                    shift = 'Day'
                    nurse_count = max(1, int(len(self.base_nurses) * self.shift_multipliers['day']))
                elif 32 <= timestep_in_day < 64:
                    shift = 'Evening'
                    nurse_count = max(1, int(len(self.base_nurses) * self.shift_multipliers['evening']))
                else:
                    shift = 'Night'
                    nurse_count = max(1, int(len(self.base_nurses) * self.shift_multipliers['night']))
                
                f.write(f"t={t:3d} | {time_str} | {shift:7s} | {nurse_count} nurses\n")

    def step(self):
        self.time += 1

        # 1. Update current nursing staff based on shift schedule
        self.nurses = self.get_current_nurses()
        
        if self.verbose and self.time % 32 == 1:  # Log shift changes (every 32 timesteps = 8 hours)
            timestep_in_day = self.time % self.timesteps_per_day
            if 0 <= timestep_in_day < 32:
                shift_name = 'day'
            elif 32 <= timestep_in_day < 64:
                shift_name = 'evening'
            else:
                shift_name = 'night'
            
            # Count carryover nurses (those completing treatments from previous shift)
            carryover_count = sum(1 for n in self.nurses if 'carryover' in str(n.id))
            new_nurses = len(self.nurses) - carryover_count
            
            print(f"[t={self.time}] Shift change: {shift_name} shift")
            print(f"    Total nurses on duty: {len(self.nurses)} ({carryover_count} carryover + {new_nurses} new)")

        # 2. New patient arrivals (use pre-generated list)
        if self.time <= len(self.patient_arrivals):
            new_patient = self.patient_arrivals[self.time - 1]
            if new_patient is not None:
                self.waiting_patients.append(new_patient)
                if self.verbose:
                    print(f"[t={self.time}] New arrival: {new_patient}")

        # 3. Update nurse status
        for nurse in self.nurses:
            if nurse.current_patient and self.time >= nurse.busy_until:
                self.completed_patients.append(nurse.current_patient)
                if self.verbose:
                    print(f"[t={self.time}] Nurse {nurse.id} finished treating {nurse.current_patient}")
                nurse.current_patient = None

        # 4. Update waiting patients
        for patient in self.waiting_patients:
            patient.wait_time += 1
            # CRITICAL FIX: Make deterioration completely deterministic per patient per timestep
            # Use patient ID + current time + base seed to create unique, deterministic seed
            # This ensures identical deterioration patterns regardless of triage order
            if patient.severity < 5:
                # Create deterministic seed unique to this patient at this timestep
                patient_time_seed = hash((patient.id, self.time, self.seed or 0)) % (2**32)
                patient_rng = random.Random(patient_time_seed)
                
                if patient_rng.random() < patient.deterioration_chance:
                    patient.severity += 1
                    # Lower deterioration chance after upgrade to avoid rapid jumps
                    patient.deterioration_chance = max(0.01, patient.deterioration_chance * 0.5)
                    # Optionally, log this event (uncomment if needed):
                    # print(f"[t={self.time}] Patient {patient.id} deteriorated to severity {patient.severity}")

        # 5. Assign free nurses (using triage policy)
        free_nurses = [n for n in self.nurses if n.current_patient is None]
        if free_nurses and self.waiting_patients:
            def triage_score(p):
                if callable(self.triage_policy):
                    return self.triage_policy(p)
                else:
                    return (
                        self.triage_policy['severity'] * p.severity +
                        self.triage_policy['deterioration'] * p.deterioration_chance +
                        self.triage_policy['wait_time'] * (p.wait_time + 1)
                    )
            self.waiting_patients.sort(key=triage_score, reverse=True)
            for nurse in free_nurses:
                if self.waiting_patients:
                    patient = self.waiting_patients.pop(0)
                    nurse.current_patient = patient
                    nurse.busy_until = self.time + patient.treatment_time
                    self.started_patients.append((patient, patient.wait_time))
                    if self.verbose:
                        print(f"[t={self.time}] Nurse {nurse.id} starts treating {patient}")

    def run(self):
        while self.time < self.total_time:
            self.step()

        # Collect metrics - Include ALL patients (treated + still waiting)
        all_patients = []
        
        # Add completed patients (those who were treated)
        if self.started_patients:
            all_patients.extend(self.started_patients)
        
        # Add patients still waiting at the end of simulation
        for patient in self.waiting_patients:
            all_patients.append((patient, patient.wait_time))
        
        if all_patients:
            waits = [wait for _, wait in all_patients]
            weighted_waits = [
                wait * p.severity * (1 + p.deterioration_chance)
                for p, wait in all_patients
            ]
            avg_wait = statistics.mean(waits)
            avg_weighted_wait = statistics.mean(weighted_waits)
            
            # Also keep track of treated patients only for comparison
            if self.started_patients:
                treated_waits = [wait for _, wait in self.started_patients]
                treated_weighted_waits = [
                    wait * p.severity * (1 + p.deterioration_chance)
                    for p, wait in self.started_patients
                ]
                avg_treated_wait = statistics.mean(treated_waits)
                avg_treated_weighted_wait = statistics.mean(treated_weighted_waits)
            else:
                avg_treated_wait = 0
                avg_treated_weighted_wait = 0
            
            # Calculate severity-specific metrics (all patients)
            severity_metrics = {}
            for severity in range(1, 6):  # Severity levels 1-5
                sev_patients = [(p, wait) for p, wait in all_patients if p.severity == severity]
                if sev_patients:
                    sev_waits = [wait for _, wait in sev_patients]
                    sev_weighted_waits = [wait * p.severity * (1 + p.deterioration_chance) for p, wait in sev_patients]
                    severity_metrics[f'sev_{severity}'] = {
                        'count': len(sev_patients),
                        'avg_wait': statistics.mean(sev_waits),
                        'avg_weighted_wait': statistics.mean(sev_weighted_waits)
                    }
                else:
                    severity_metrics[f'sev_{severity}'] = {
                        'count': 0,
                        'avg_wait': 0,
                        'avg_weighted_wait': 0
                    }
            
            return {
                'completed': len(self.completed_patients),
                'still_waiting': len(self.waiting_patients),
                'avg_wait': avg_wait,  # All patients (treated + waiting)
                'max_wait': max(waits),
                'avg_weighted_wait': avg_weighted_wait,  # All patients (treated + waiting)
                'avg_treated_wait': avg_treated_wait,  # Only treated patients
                'avg_treated_weighted_wait': avg_treated_weighted_wait,  # Only treated patients
                'total_patients': len(all_patients),
                'severity_metrics': severity_metrics
            }
        else:
            # No patients at all - initialize empty severity metrics
            severity_metrics = {}
            for severity in range(1, 6):
                severity_metrics[f'sev_{severity}'] = {
                    'count': 0,
                    'avg_wait': 0,
                    'avg_weighted_wait': 0
                }
            
            return {
                'completed': 0,
                'still_waiting': len(self.waiting_patients),
                'avg_wait': None,
                'max_wait': None,
                'avg_weighted_wait': None,
                'avg_treated_wait': None,
                'avg_treated_weighted_wait': None,
                'total_patients': 0,
                'severity_metrics': severity_metrics
            }



# This file now contains only the core classes - Patient, Nurse, and ERSimulation
# Other functionality has been moved to separate modules for better organization
