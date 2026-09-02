from typing import Dict, Any, List, Tuple

class ConstraintValidator:
    def __init__(self, constraints: Dict[str, Any]):
        self.constraints = constraints

    def validate(self, instance: Dict[str, Any]) -> bool:
        is_valid, _ = self.validate_full(instance)
        return is_valid

    def validate_full(self, instance: Dict[str, Any]) -> Tuple[bool, List[str]]:
        violations = []

        # 1. Attribute Domain Checks
        domains = self.constraints.get("attribute_domain", {})
        for attr, specs in domains.items():
            if attr not in instance:
                violations.append(f"Missing attribute: {attr}")
                continue
            
            val = instance[attr]
            
            if specs["type"] == "numerical":
                try:
                    val_float = float(val)
                except ValueError:
                    violations.append(f"Attribute {attr} must be numerical, got {val}")
                    continue
                    
                if not (specs["min"] <= val_float <= specs["max"]):
                    violations.append(f"Attribute {attr} value {val} out of range [{specs['min']}, {specs['max']}]")
            
            elif specs["type"] == "categorical":
                if str(val) not in specs["values"]:
                     violations.append(f"Attribute {attr} value '{val}' not in allowed categories")

        # 2. Inter-Attribute Dependency Checks
        dependencies = self.constraints.get("inter_attribute", [])
        for rule in dependencies:
            if rule["type"] == "dependency_exclusion":
                ant = rule["antecedent"]
                cons = rule["consequent_exclusion"]
                
                # Check if antecedent applies
                antecedent_met = False
                
                if "range" in ant:
                    # Interval antecedent — qcut produces (low, high]
                    val = instance.get(ant["attribute"])
                    if val is not None:
                        try:
                            v_float = float(val)
                            if ant["range"]["min"] < v_float <= ant["range"]["max"]:
                                antecedent_met = True
                        except (ValueError, TypeError):
                            pass
                elif "sentinel_value" in ant:
                    # Sentinel antecedent — numeric equality (e.g. pdays == -1)
                    val = instance.get(ant["attribute"])
                    if val is not None:
                        try:
                            if float(val) == float(ant["sentinel_value"]):
                                antecedent_met = True
                        except (ValueError, TypeError):
                            pass
                else:
                    # Categorical value antecedent — string equality
                    if str(instance.get(ant["attribute"])) == str(ant["value"]):
                        antecedent_met = True
                
                if antecedent_met:
                    # Check if consequent violates
                    actual_val = str(instance.get(cons["attribute"]))
                    if actual_val in cons["values"]:
                        violations.append(
                            f"Dependency violation: {ant['attribute']} condition met "
                            f"implies {cons['attribute']} cannot be {actual_val}"
                        )

        # 3. Structural Checks
        structural = self.constraints.get("structural", [])
        for rule in structural:
            if rule["type"] == "non_null":
                if instance.get(rule["attribute"]) is None:
                    violations.append(f"Structural violation: Attribute {rule['attribute']} cannot be None")

        return len(violations) == 0, violations
