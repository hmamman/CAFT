import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/caft')))

import pandas as pd
import numpy as np
from constraint_extractor import DatasetConstraintExtractor
from validator import ConstraintValidator

def run_test():
    print("=== Generative Constraint Framework Test ===")
    
    # 1. Create Dummy Dataset (Adult-like)
    np.random.seed(42)
    n = 1000
    data = {
        "age": np.random.randint(17, 70, n),
        "education": np.random.choice(["HS-grad", "Bachelors", "Masters"], n),
        "hours_per_week": np.random.randint(1, 60, n),
    }
    df = pd.DataFrame(data)
    
    # Inject Critical Dependency: Age < 22 => Education != Masters
    # (To test the binning extractor)
    df.loc[df["age"] < 22, "education"] = "HS-grad"
    
    print("Dataset Created. Head:")
    print(df.head())
    
    # 2. Extract Constraints
    print("\nExtracting Constraints...")
    extractor = DatasetConstraintExtractor(df, n_bins=10, min_confidence=0.99)
    # Force binning logic to run
    extractor.extract_attribute_domain_constraints()
    extractor.extract_inter_attribute_constraints() # This triggers the new logic
    extractor.extract_structural_constraints()
    
    specs = extractor.constraints
    print(f"Constraints Extracted: {len(specs['inter_attribute'])} inter-attribute rules found.")
    for r in specs['inter_attribute']:
        print(r)
    
    # Verify we found the age->education rule
    found_rule = False
    for rule in specs["inter_attribute"]:
        if rule.get("type") == "dependency_exclusion":
            if rule["antecedent"]["attribute"] == "binned_age" and \
               rule["consequent_exclusion"]["attribute"] == "education":
                   print(f"FAILED RULE FOUND: {rule}") 
                   # Wait, 'binned_age' is an internal name, let's see what the extractor produces.
                   # It produces keys like "binned_age".
                   found_rule = True
    
    # 3. Validation Test
    print("\nValidating Instances...")
    validator = ConstraintValidator(specs)
    
    # Case A: Valid Instance
    valid_instance = {"age": 30, "education": "Masters", "hours_per_week": 40}
    # Note: 'binned_age' is NOT in the instance. The validator currently checks raw values.
    # PROBLEM: The extractor writes rules using 'binned_age', but the instance has raw 'age'.
    # The validator needs to know how to bin 'age' to check the rule.
    # OR the extractor should express the rule in terms of raw ranges.
    
    # CURRENT ARCHITECTURE GAP: Validator doesn't know how to bin raw values to check binned rules.
    # We need to fix this in Validator or Extractor.
    # For now, let's keep the script observing this behavior.
    
    v, reasons = validator.validate_full(valid_instance)
    print(f"Valid Instance Test: {v}, Reasons: {reasons}")

    # Case B: Invalid Instance (Violating the Age < 22 -> Education != Masters rule)
    print("\nInvalid Instance Test (Age=19, Education='Masters')...")
    invalid_instance = {"age": 19, "education": "Masters", "hours_per_week": 40}
    v_inv, reasons_inv = validator.validate_full(invalid_instance)
    print(f"Invalid Instance Test: {v_inv}, Reasons: {reasons_inv}")
    
    if not v_inv and any("Masters" in r for r in reasons_inv):
        print("SUCCESS: Validator correctly caught the dependency violation.")
    else:
        print("FAILURE: Validator failed to catch the violation.")

if __name__ == "__main__":
    run_test()
