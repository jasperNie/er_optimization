#!/usr/bin/env python3
"""
Main evaluation script for ER Optimization algorithms
Run this to compare all algorithms against ESI and MTS baselines
"""

import sys
import os
sys.path.append('evaluations')

from enhanced_evaluation import main

if __name__ == "__main__":
    main()