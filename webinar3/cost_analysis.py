"""
Cost-Benefit Analysis Tools
===========================
Analyze trade-offs between optimization scenarios.

Part of Webinar 3: Optimizing Multilingual NLP Models
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class OptimizationScenario:
    """Model an optimization scenario with costs and benefits."""
    name: str
    accuracy_by_language: Dict[str, float]
    latency_p95_ms: float
    monthly_infra_cost: float
    model_size_mb: float


class CostBenefitAnalyzer:
    """
    Analyze cost-benefit tradeoffs of optimization scenarios.
    
    Integrates:
    - Infrastructure costs
    - Error costs per language
    - Traffic distribution
    - Business value per language
    """
    
    def __init__(
        self,
        traffic_distribution: Dict[str, float],
        error_cost_per_language: Dict[str, float],
        monthly_predictions: int = 10_000_000
    ):
        self.traffic_distribution = traffic_distribution
        self.error_costs = error_cost_per_language
        self.monthly_predictions = monthly_predictions
    
    def calculate_total_monthly_cost(self, scenario: OptimizationScenario) -> Dict:
        """
        Calculate total monthly cost including infrastructure and error costs.
        """
        # Infrastructure cost
        infra_cost = scenario.monthly_infra_cost
        
        # Error cost per language
        error_costs_by_lang = {}
        total_error_cost = 0
        
        for lang, accuracy in scenario.accuracy_by_language.items():
            traffic_share = self.traffic_distribution.get(lang, 0)
            error_cost = self.error_costs.get(lang, 0)
            
            predictions_for_lang = self.monthly_predictions * traffic_share
            errors_for_lang = predictions_for_lang * (1 - accuracy)
            cost_for_lang = errors_for_lang * error_cost
            
            error_costs_by_lang[lang] = cost_for_lang
            total_error_cost += cost_for_lang
        
        return {
            "scenario_name": scenario.name,
            "infrastructure_cost": infra_cost,
            "total_error_cost": total_error_cost,
            "error_cost_by_language": error_costs_by_lang,
            "total_monthly_cost": infra_cost + total_error_cost,
            "latency_p95_ms": scenario.latency_p95_ms,
            "model_size_mb": scenario.model_size_mb
        }
    
    def compare_scenarios(
        self,
        scenarios: List[OptimizationScenario]
    ) -> List[Dict]:
        """Compare multiple optimization scenarios."""
        results = []
        
        for scenario in scenarios:
            result = self.calculate_total_monthly_cost(scenario)
            results.append(result)
        
        # Sort by total cost
        results.sort(key=lambda x: x['total_monthly_cost'])
        
        return results
    
    def find_pareto_optimal(
        self,
        scenarios: List[OptimizationScenario]
    ) -> List[str]:
        """
        Find Pareto-optimal scenarios (no scenario dominates on all dimensions).
        
        Dimensions: total cost, latency, overall accuracy
        """
        scenario_metrics = []
        
        for s in scenarios:
            cost = self.calculate_total_monthly_cost(s)['total_monthly_cost']
            overall_acc = sum(
                s.accuracy_by_language[lang] * self.traffic_distribution.get(lang, 0)
                for lang in s.accuracy_by_language
            )
            scenario_metrics.append({
                'name': s.name,
                'cost': cost,
                'latency': s.latency_p95_ms,
                'accuracy': overall_acc
            })
        
        # Find Pareto frontier
        pareto_optimal = []
        
        for s in scenario_metrics:
            is_dominated = False
            
            for other in scenario_metrics:
                if other['name'] == s['name']:
                    continue
                
                # Check if 'other' dominates 's'
                if (other['cost'] <= s['cost'] and 
                    other['latency'] <= s['latency'] and 
                    other['accuracy'] >= s['accuracy'] and
                    (other['cost'] < s['cost'] or 
                     other['latency'] < s['latency'] or 
                     other['accuracy'] > s['accuracy'])):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(s['name'])
        
        return pareto_optimal


def example_cost_benefit_analysis():
    """Demonstrate cost-benefit analysis for optimization decisions."""
    
    # Setup
    traffic = {'en': 0.70, 'es': 0.15, 'ru': 0.10, 'th': 0.05}
    error_costs = {'en': 10, 'es': 15, 'ru': 20, 'th': 50}  # USD per error
    
    analyzer = CostBenefitAnalyzer(
        traffic_distribution=traffic,
        error_cost_per_language=error_costs,
        monthly_predictions=1_000_000
    )
    
    # Define scenarios
    scenarios = [
        OptimizationScenario(
            name="Baseline (XLM-R Large)",
            accuracy_by_language={'en': 0.92, 'es': 0.88, 'ru': 0.84, 'th': 0.78},
            latency_p95_ms=180,
            monthly_infra_cost=5000,
            model_size_mb=2200
        ),
        OptimizationScenario(
            name="Quantized",
            accuracy_by_language={'en': 0.91, 'es': 0.86, 'ru': 0.79, 'th': 0.71},
            latency_p95_ms=90,
            monthly_infra_cost=2000,
            model_size_mb=550
        ),
        OptimizationScenario(
            name="Distilled",
            accuracy_by_language={'en': 0.90, 'es': 0.85, 'ru': 0.81, 'th': 0.74},
            latency_p95_ms=55,
            monthly_infra_cost=1500,
            model_size_mb=500
        ),
        OptimizationScenario(
            name="Distilled + Adapters",
            accuracy_by_language={'en': 0.91, 'es': 0.86, 'ru': 0.83, 'th': 0.76},
            latency_p95_ms=68,
            monthly_infra_cost=2000,
            model_size_mb=550
        )
    ]
    
    # Compare
    results = analyzer.compare_scenarios(scenarios)
    pareto = analyzer.find_pareto_optimal(scenarios)
    
    print("COST-BENEFIT ANALYSIS RESULTS")
    print("=" * 60)
    
    for r in results:
        print(f"\n{r['scenario_name']}")
        print(f"  Infrastructure:  ${r['infrastructure_cost']:,.0f}/month")
        print(f"  Error Costs:     ${r['total_error_cost']:,.0f}/month")
        print(f"  TOTAL:           ${r['total_monthly_cost']:,.0f}/month")
        print(f"  Latency (p95):   {r['latency_p95_ms']}ms")
    
    print(f"\nPareto-optimal scenarios: {pareto}")


if __name__ == "__main__":
    example_cost_benefit_analysis()
