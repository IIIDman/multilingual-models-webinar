"""
A/B Testing Utilities
=====================
Traffic routing for model experiments.
"""

import hashlib
from typing import Literal, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ExperimentConfig:
    """Configuration for an A/B test experiment."""
    name: str
    control_weight: float  # 0.0 to 1.0
    treatment_name: str = "treatment"
    control_name: str = "control"
    description: str = ""
    created_at: str = ""
    
    def __post_init__(self):
        if not 0.0 <= self.control_weight <= 1.0:
            raise ValueError("control_weight must be between 0.0 and 1.0")
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class ABTestRouter:
    """
    Route traffic between control (old) and treatment (new) models.
    
    Uses deterministic hashing so the same request_id always gets 
    the same variant (important for user experience consistency).
    
    Example:
        >>> router = ABTestRouter('model_v2_experiment', control_weight=0.95)
        >>> variant = router.get_variant('user_123')
        >>> if variant == 'treatment':
        ...     result = new_model.predict(text)
        ... else:
        ...     result = old_model.predict(text)
    """
    
    def __init__(
        self, 
        experiment_name: str, 
        control_weight: float = 0.95,
        control_name: str = "control",
        treatment_name: str = "treatment"
    ):
        """
        Initialize A/B test router.
        
        Args:
            experiment_name: Unique name for this experiment
            control_weight: Fraction of traffic to control (e.g., 0.95 = 95%)
            control_name: Name for control variant
            treatment_name: Name for treatment variant
        """
        self.config = ExperimentConfig(
            name=experiment_name,
            control_weight=control_weight,
            control_name=control_name,
            treatment_name=treatment_name
        )
        self._stats = {control_name: 0, treatment_name: 0}
    
    def get_variant(self, request_id: str) -> str:
        """
        Deterministically assign a request to control or treatment.
        
        Same request_id always gets same variant.
        
        Args:
            request_id: Unique identifier (user ID, session ID, request ID)
            
        Returns:
            Variant name ('control' or 'treatment')
        """
        # Create deterministic hash
        hash_input = f"{self.config.name}:{request_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000.0  # 0.0 to 0.9999
        
        if bucket < self.config.control_weight:
            variant = self.config.control_name
        else:
            variant = self.config.treatment_name
        
        self._stats[variant] += 1
        return variant
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total = sum(self._stats.values())
        return {
            'experiment': self.config.name,
            'control_weight': self.config.control_weight,
            'total_requests': total,
            'variants': {
                name: {
                    'count': count,
                    'actual_pct': count / total if total > 0 else 0
                }
                for name, count in self._stats.items()
            }
        }
    
    def update_weight(self, new_control_weight: float) -> None:
        """
        Update the traffic split.
        
        Use this to gradually increase treatment traffic.
        
        Args:
            new_control_weight: New control weight (0.0 to 1.0)
        """
        if not 0.0 <= new_control_weight <= 1.0:
            raise ValueError("control_weight must be between 0.0 and 1.0")
        
        old_weight = self.config.control_weight
        self.config.control_weight = new_control_weight
        print(f"Updated {self.config.name}: {old_weight:.1%} -> {new_control_weight:.1%} control")


class MultiVariantRouter:
    """
    Route traffic between multiple variants (not just A/B).
    
    Example:
        >>> router = MultiVariantRouter('model_comparison')
        >>> router.add_variant('baseline', weight=0.70)
        >>> router.add_variant('model_v2', weight=0.15)
        >>> router.add_variant('model_v3', weight=0.15)
        >>> variant = router.get_variant('user_123')
    """
    
    def __init__(self, experiment_name: str):
        """Initialize multi-variant router."""
        self.experiment_name = experiment_name
        self.variants: Dict[str, float] = {}
        self._stats: Dict[str, int] = {}
    
    def add_variant(self, name: str, weight: float) -> None:
        """
        Add a variant with specified weight.
        
        Args:
            name: Variant name
            weight: Traffic weight (0.0 to 1.0)
        """
        self.variants[name] = weight
        self._stats[name] = 0
        
        total_weight = sum(self.variants.values())
        if total_weight > 1.0:
            print(f"Warning: Total weight ({total_weight:.2f}) exceeds 1.0")
    
    def get_variant(self, request_id: str) -> str:
        """Get variant for a request."""
        if not self.variants:
            raise ValueError("No variants configured")
        
        hash_input = f"{self.experiment_name}:{request_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000.0
        
        cumulative = 0.0
        for name, weight in self.variants.items():
            cumulative += weight
            if bucket < cumulative:
                self._stats[name] += 1
                return name
        
        # Fallback to last variant
        last_variant = list(self.variants.keys())[-1]
        self._stats[last_variant] += 1
        return last_variant


def gradual_rollout_schedule(
    start_pct: float = 0.05,
    end_pct: float = 1.0,
    days: int = 7
) -> Dict[int, float]:
    """
    Generate a gradual rollout schedule.
    
    Args:
        start_pct: Starting treatment percentage
        end_pct: Final treatment percentage
        days: Number of days for rollout
        
    Returns:
        Dictionary mapping day number to treatment percentage
    """
    schedule = {}
    step = (end_pct - start_pct) / (days - 1) if days > 1 else 0
    
    for day in range(days):
        schedule[day] = min(start_pct + (step * day), end_pct)
    
    return schedule


# Example usage
if __name__ == "__main__":
    print("A/B Testing Example")
    print("=" * 40)
    
    # Create router with 95% control, 5% treatment
    router = ABTestRouter('multilingual_model_v2', control_weight=0.95)
    
    # Simulate traffic
    for i in range(1000):
        router.get_variant(f"user_{i}")
    
    stats = router.get_stats()
    print(f"\nExperiment: {stats['experiment']}")
    print(f"Target control weight: {stats['control_weight']:.1%}")
    print(f"Total requests: {stats['total_requests']}")
    for variant, data in stats['variants'].items():
        print(f"  {variant}: {data['count']} ({data['actual_pct']:.1%})")
    
    # Demonstrate deterministic routing
    print("\nDeterministic routing test:")
    user_id = "user_12345"
    results = [router.get_variant(user_id) for _ in range(5)]
    print(f"  Same user '{user_id}' always gets: {results[0]} (verified: {len(set(results)) == 1})")
    
    # Show gradual rollout schedule
    print("\nGradual rollout schedule (7 days):")
    schedule = gradual_rollout_schedule(start_pct=0.05, end_pct=1.0, days=7)
    for day, pct in schedule.items():
        print(f"  Day {day}: {pct:.1%} treatment traffic")
