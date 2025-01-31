from .base_agent import BaseAgent
from typing import Dict, Any

class CostAgent(BaseAgent):
    """Agent responsible for cost analysis and variance detection."""
    
    def __init__(self):
        super().__init__("Cost Analysis Agent")
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process cost data and return analysis results.
        
        Args:
            data: Dictionary containing cost data
            
        Returns:
            Dictionary containing analysis results and anomalies
        """
        anomalies = self.analyze_data(data)
        
        return {
            "anomalies": anomalies,
            "metrics": self._calculate_metrics(data),
            "status": "completed"
        }
        
    def _calculate_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate cost-related metrics from the data."""
        metrics = {
            "total_planned_cost": 0,
            "total_actual_cost": 0,
            "average_variance": 0,
            "variance_count": 0
        }
        
        if 'costs' in data:
            for cost_item in data['costs']:
                if 'planned' in cost_item:
                    metrics['total_planned_cost'] += cost_item['planned']
                if 'actual' in cost_item:
                    metrics['total_actual_cost'] += cost_item['actual']
                if 'planned' in cost_item and 'actual' in cost_item:
                    variance = abs(cost_item['actual'] - cost_item['planned'])
                    metrics['average_variance'] += variance
                    metrics['variance_count'] += 1
                    
        if metrics['variance_count'] > 0:
            metrics['average_variance'] /= metrics['variance_count']
            
        return metrics
        
    def analyze_data(self, data):
        """
        Analyze cost data for anomalies and variances.
        
        Args:
            data: Dictionary containing cost data
            
        Returns:
            List of dictionaries containing detected anomalies
        """
        anomalies = []
        
        if not data:
            return anomalies
            
        # Check for cost variances
        if 'costs' in data:
            for line_num, cost_item in enumerate(data['costs'], 1):
                # Check for negative costs
                if cost_item.get('amount', 0) < 0:
                    anomalies.append({
                        'type': 'Negative Cost',
                        'description': f'Negative cost amount detected: {cost_item["amount"]}',
                        'line_number': line_num,
                        'severity': 'High',
                        'recommendation': 'Review and verify the negative cost entry'
                    })
                
                # Check for unusually high variances
                if 'planned' in cost_item and 'actual' in cost_item:
                    variance = abs(cost_item['actual'] - cost_item['planned'])
                    variance_pct = variance / cost_item['planned'] if cost_item['planned'] != 0 else float('inf')
                    
                    if variance_pct > 0.20:  # More than 20% variance
                        anomalies.append({
                            'type': 'Cost Variance',
                            'description': f'High cost variance detected: {variance_pct:.1%}',
                            'line_number': line_num,
                            'severity': 'Medium',
                            'recommendation': 'Investigate cause of significant cost variance'
                        })
        
        return anomalies
