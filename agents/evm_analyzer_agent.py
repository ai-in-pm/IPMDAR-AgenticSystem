from typing import Dict, Any
from .base_agent import BaseAgent

class EVMAnalyzerAgent(BaseAgent):
    def __init__(self, user_role: str = "analyst"):
        super().__init__(user_role)
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes cost & schedule performance
        
        Args:
            data: Processed IPMDAR data containing EVM metrics
            
        Returns:
            Dict containing EVM analysis results
        """
        if not self.check_access("analyze"):
            return {"error": "Unauthorized access"}
            
        problem = "Analyzing EVM metrics and performance indicators"
        process_steps = [
            "Step 1: Checking user authorization",
            "Step 2: Extracting CPI and SPI metrics",
            "Step 3: Analyzing performance trends",
            "Step 4: Generating recommendations"
        ]
        
        try:
            # Extract EVM metrics
            cpi = data.get("CPI", 1)
            spi = data.get("SPI", 1)
            
            # Analyze performance
            performance_status = self._analyze_performance(cpi, spi)
            recommendations = self._generate_recommendations(cpi, spi)
            
            justification = f"Analysis based on CPI ({cpi}) and SPI ({spi}) values"
            output = {
                "status": performance_status["status"],
                "message": performance_status["message"],
                "metrics": {
                    "CPI": cpi,
                    "SPI": spi
                },
                "recommendations": recommendations
            }
            
            # Record Chain of Thought
            cot_record = self.chain_of_thought(problem, process_steps, justification, output)
            output["reasoning"] = cot_record
            
            return output
            
        except Exception as e:
            self.logger.error(f"EVM Analysis Error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
            
    def _analyze_performance(self, cpi: float, spi: float) -> Dict[str, str]:
        """Analyze performance based on CPI and SPI"""
        if cpi < 0.85 or spi < 0.85:
            return {
                "status": "critical",
                "message": "Severe performance issues detected. Immediate action required."
            }
        elif cpi < 1 or spi < 1:
            return {
                "status": "warning",
                "message": "Performance declining. Corrective action recommended."
            }
        else:
            return {
                "status": "good",
                "message": "Project performing well."
            }
            
    def _generate_recommendations(self, cpi: float, spi: float) -> list:
        """Generate recommendations based on performance metrics"""
        recommendations = []
        
        if cpi < 1:
            recommendations.extend([
                "Review cost control measures",
                "Identify cost overrun sources",
                "Implement cost reduction strategies"
            ])
            
        if spi < 1:
            recommendations.extend([
                "Assess schedule delays",
                "Optimize resource allocation",
                "Consider schedule recovery options"
            ])
            
        if not recommendations:
            recommendations.append("Continue current performance monitoring")
            
        return recommendations
