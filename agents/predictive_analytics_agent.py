from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from .base_agent import BaseAgent

class PredictiveAnalyticsAgent(BaseAgent):
    def __init__(self, user_role: str = "analyst"):
        super().__init__(user_role)
        self.scaler = StandardScaler()
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs predictive analytics and forecasting
        
        Args:
            data: Dictionary containing EVM metrics and historical data
            
        Returns:
            Dict containing forecasts and predictions
        """
        if not self.check_access("analyze"):
            return {"error": "Unauthorized access"}
            
        problem = "Forecasting project performance and completion"
        process_steps = [
            "Step 1: Checking user authorization",
            "Step 2: Preparing historical data",
            "Step 3: Training predictive models",
            "Step 4: Generating forecasts",
            "Step 5: Computing confidence intervals",
            "Step 6: Formulating recommendations"
        ]
        
        try:
            # Extract historical data
            historical_data = self._prepare_historical_data(data)
            
            # Generate forecasts
            cost_forecast = self._forecast_cost(historical_data)
            schedule_forecast = self._forecast_schedule(historical_data)
            
            # Compute completion estimates
            completion_estimate = self._estimate_completion(
                cost_forecast,
                schedule_forecast,
                historical_data
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                cost_forecast,
                schedule_forecast,
                completion_estimate
            )
            
            justification = f"Forecasts based on {len(historical_data.get('periods', []))} " \
                           f"periods of historical data with {completion_estimate['confidence_level']:.1%} " \
                           f"confidence in completion estimates"
                           
            output = {
                "status": completion_estimate["status"],
                "message": completion_estimate["message"],
                "cost_forecast": cost_forecast,
                "schedule_forecast": schedule_forecast,
                "completion_estimate": completion_estimate,
                "recommendations": recommendations
            }
            
            # Record Chain of Thought
            cot_record = self.chain_of_thought(problem, process_steps, justification, output)
            output["reasoning"] = cot_record
            
            return output
            
        except Exception as e:
            self.logger.error(f"Predictive Analytics Error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
            
    def _prepare_historical_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare historical data for analysis"""
        historical_data = {
            "periods": [],
            "cpi_trend": [],
            "spi_trend": [],
            "cost_variance": [],
            "schedule_variance": []
        }
        
        if "historical_performance" in data:
            hist_perf = data["historical_performance"]
            for period in sorted(hist_perf.keys()):
                historical_data["periods"].append(float(period))
                historical_data["cpi_trend"].append(hist_perf[period].get("CPI", 1.0))
                historical_data["spi_trend"].append(hist_perf[period].get("SPI", 1.0))
                historical_data["cost_variance"].append(hist_perf[period].get("CV", 0.0))
                historical_data["schedule_variance"].append(hist_perf[period].get("SV", 0.0))
                
        return historical_data
        
    def _forecast_cost(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cost forecasts"""
        if not historical_data["periods"]:
            return {"status": "error", "message": "Insufficient historical data"}
            
        X = np.array(historical_data["periods"]).reshape(-1, 1)
        y_cpi = np.array(historical_data["cpi_trend"])
        y_cv = np.array(historical_data["cost_variance"])
        
        # Train models
        cpi_model = LinearRegression().fit(X, y_cpi)
        cv_model = LinearRegression().fit(X, y_cv)
        
        # Generate forecasts
        next_period = max(historical_data["periods"]) + 1
        forecast_periods = np.array([next_period, next_period + 1, next_period + 2]).reshape(-1, 1)
        
        cpi_forecast = cpi_model.predict(forecast_periods)
        cv_forecast = cv_model.predict(forecast_periods)
        
        return {
            "periods": forecast_periods.flatten().tolist(),
            "cpi_forecast": cpi_forecast.tolist(),
            "cv_forecast": cv_forecast.tolist(),
            "trend": "improving" if cpi_model.coef_[0] > 0 else "declining"
        }
        
    def _forecast_schedule(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate schedule forecasts"""
        if not historical_data["periods"]:
            return {"status": "error", "message": "Insufficient historical data"}
            
        X = np.array(historical_data["periods"]).reshape(-1, 1)
        y_spi = np.array(historical_data["spi_trend"])
        y_sv = np.array(historical_data["schedule_variance"])
        
        # Train models
        spi_model = LinearRegression().fit(X, y_spi)
        sv_model = LinearRegression().fit(X, y_sv)
        
        # Generate forecasts
        next_period = max(historical_data["periods"]) + 1
        forecast_periods = np.array([next_period, next_period + 1, next_period + 2]).reshape(-1, 1)
        
        spi_forecast = spi_model.predict(forecast_periods)
        sv_forecast = sv_model.predict(forecast_periods)
        
        return {
            "periods": forecast_periods.flatten().tolist(),
            "spi_forecast": spi_forecast.tolist(),
            "sv_forecast": sv_forecast.tolist(),
            "trend": "improving" if spi_model.coef_[0] > 0 else "declining"
        }
        
    def _estimate_completion(self, cost_forecast: Dict[str, Any],
                           schedule_forecast: Dict[str, Any],
                           historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate project completion based on forecasts"""
        # Calculate confidence level based on forecast stability
        confidence_level = self._calculate_confidence_level(
            cost_forecast,
            schedule_forecast,
            historical_data
        )
        
        # Determine status based on forecasts
        if cost_forecast["trend"] == "declining" and schedule_forecast["trend"] == "declining":
            status = "critical"
            message = "Project trending towards significant overruns"
        elif cost_forecast["trend"] == "declining" or schedule_forecast["trend"] == "declining":
            status = "warning"
            message = "Project showing signs of performance decline"
        else:
            status = "good"
            message = "Project trending towards successful completion"
            
        return {
            "status": status,
            "message": message,
            "confidence_level": confidence_level,
            "estimated_completion_period": self._calculate_completion_period(
                schedule_forecast,
                historical_data
            )
        }
        
    def _calculate_confidence_level(self, cost_forecast: Dict[str, Any],
                                  schedule_forecast: Dict[str, Any],
                                  historical_data: Dict[str, Any]) -> float:
        """Calculate confidence level in forecasts"""
        # Simple confidence calculation based on trend stability
        cpi_stability = np.std(historical_data["cpi_trend"]) if historical_data["cpi_trend"] else 1
        spi_stability = np.std(historical_data["spi_trend"]) if historical_data["spi_trend"] else 1
        
        confidence = 1.0 - (cpi_stability + spi_stability) / 4
        return max(0.0, min(1.0, confidence))
        
    def _calculate_completion_period(self, schedule_forecast: Dict[str, Any],
                                   historical_data: Dict[str, Any]) -> int:
        """Calculate estimated completion period"""
        if schedule_forecast.get("spi_forecast"):
            latest_spi = schedule_forecast["spi_forecast"][-1]
            remaining_periods = int(10 / latest_spi) if latest_spi > 0 else 20
            return max(historical_data["periods"]) + remaining_periods
        return max(historical_data["periods"]) + 10
        
    def _generate_recommendations(self, cost_forecast: Dict[str, Any],
                                schedule_forecast: Dict[str, Any],
                                completion_estimate: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on forecasts"""
        recommendations = []
        
        # Cost-related recommendations
        if cost_forecast["trend"] == "declining":
            recommendations.extend([
                "Implement cost control measures",
                "Review resource allocation efficiency",
                "Identify and address cost overrun sources"
            ])
            
        # Schedule-related recommendations
        if schedule_forecast["trend"] == "declining":
            recommendations.extend([
                "Accelerate critical path activities",
                "Review and optimize resource assignments",
                "Consider schedule compression techniques"
            ])
            
        # Completion-related recommendations
        if completion_estimate["status"] == "critical":
            recommendations.extend([
                "Initiate project recovery plan",
                "Escalate to senior management",
                "Consider scope adjustment options"
            ])
        elif completion_estimate["status"] == "warning":
            recommendations.extend([
                "Increase monitoring frequency",
                "Prepare contingency plans",
                "Review risk mitigation strategies"
            ])
            
        if not recommendations:
            recommendations.append("Maintain current project controls")
            
        return recommendations
