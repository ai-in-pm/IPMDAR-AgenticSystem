import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from .base_agent import BaseAgent

class ScheduleRiskAgent(BaseAgent):
    def __init__(self, user_role: str = "analyst"):
        super().__init__(user_role)
        self.model = None
        self.scaler = StandardScaler()
        
    def _extract_temporal_features(self, baseline_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from schedule data"""
        features = pd.DataFrame()
        
        # Duration features
        features['baseline_duration'] = baseline_df['duration']
        features['actual_duration'] = current_df['duration']
        features['duration_variance'] = features['actual_duration'] - features['baseline_duration']
        features['duration_variance_pct'] = (features['duration_variance'] / features['baseline_duration']) * 100
        
        # Critical path features
        features['baseline_cp_length'] = baseline_df['critical_path_length']
        features['current_cp_length'] = current_df['critical_path_length']
        features['cp_length_variance'] = features['current_cp_length'] - features['baseline_cp_length']
        features['cp_length_variance_pct'] = (features['cp_length_variance'] / features['baseline_cp_length']) * 100
        
        # Start date variance (in days)
        features['start_variance'] = (pd.to_datetime(current_df['actual_start']) - 
                                    pd.to_datetime(baseline_df['planned_start'])).dt.days
        
        # Finish date variance (in days)
        features['finish_variance'] = (pd.to_datetime(current_df['forecast_finish']) - 
                                     pd.to_datetime(baseline_df['planned_finish'])).dt.days
        
        # Schedule performance features
        features['schedule_performance_index'] = features['baseline_duration'] / features['actual_duration']
        features['critical_ratio'] = 1 / features['schedule_performance_index']
        
        # Float consumption
        features['total_float_consumed'] = features['duration_variance']
        features['float_consumption_rate'] = features['total_float_consumed'] / features['baseline_duration']
        
        # Risk indicators
        features['is_delayed_start'] = (features['start_variance'] > 0).astype(int)
        features['is_extended_duration'] = (features['duration_variance'] > 0).astype(int)
        features['is_critical_delay'] = ((features['finish_variance'] > 0) & 
                                       (features['schedule_performance_index'] < 0.9)).astype(int)
        features['is_cp_extended'] = (features['cp_length_variance'] > 0).astype(int)
        
        return features
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
        """Train the schedule risk assessment model"""
        try:
            # Initialize model with optimized parameters
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            return self.model
            
        except Exception as e:
            self.logger.error(f"Error training schedule risk model: {e}")
            raise
            
    def predict_risk(self, baseline_schedule: Dict[str, Any], current_schedule: Dict[str, Any]) -> Dict[str, Any]:
        """Predict schedule risk level and provide detailed analysis"""
        try:
            # Convert input to DataFrames
            baseline_df = pd.DataFrame([baseline_schedule])
            current_df = pd.DataFrame([current_schedule])
            
            # Extract features
            features = self._extract_temporal_features(baseline_df, current_df)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict risk level
            risk_level = self.model.predict(features_scaled)[0]
            risk_probs = self.model.predict_proba(features_scaled)[0]
            
            # Get feature importances
            feature_importance = dict(zip(features.columns, 
                                       self.model.feature_importances_))
            
            return {
                "risk_level": risk_level,
                "risk_probabilities": {
                    "low": float(risk_probs[0]),
                    "medium": float(risk_probs[1]),
                    "high": float(risk_probs[2])
                },
                "key_factors": {
                    k: float(v) for k, v in sorted(
                        feature_importance.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:5]
                },
                "metrics": {
                    "duration_variance": float(features['duration_variance'].iloc[0]),
                    "schedule_performance_index": float(features['schedule_performance_index'].iloc[0]),
                    "float_consumption_rate": float(features['float_consumption_rate'].iloc[0])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting schedule risk: {e}")
            return {
                "error": str(e)
            }
            
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs Monte Carlo schedule risk analysis
        
        Args:
            data: Processed IPMDAR data containing schedule information
            
        Returns:
            Dict containing risk analysis results
        """
        if not self.check_access("analyze"):
            return {"error": "Unauthorized access"}
            
        problem = "Evaluating schedule risks and performing Monte Carlo simulation"
        process_steps = [
            "Step 1: Checking user authorization",
            "Step 2: Extracting schedule data",
            "Step 3: Running Monte Carlo simulation",
            "Step 4: Computing risk scores",
            "Step 5: Generating risk mitigation recommendations"
        ]
        
        try:
            # Extract schedule data
            schedule_data = data.get("schedule_data", {})
            
            # Perform Monte Carlo simulation
            simulation_results = self._run_monte_carlo_simulation(schedule_data)
            risk_score = simulation_results["risk_score"]
            
            # Generate risk assessment
            risk_assessment = self._assess_risk(risk_score)
            recommendations = self._generate_recommendations(risk_assessment["risk_level"], simulation_results)
            
            justification = f"Risk assessment based on Monte Carlo simulation with risk score: {risk_score:.2f}"
            output = {
                "status": risk_assessment["status"],
                "risk_level": risk_assessment["risk_level"],
                "message": risk_assessment["message"],
                "confidence_level": simulation_results["confidence_level"],
                "risk_factors": simulation_results["risk_factors"],
                "recommendations": recommendations
            }
            
            # Record Chain of Thought
            cot_record = self.chain_of_thought(problem, process_steps, justification, output)
            output["reasoning"] = cot_record
            
            return output
            
        except Exception as e:
            self.logger.error(f"Schedule Risk Analysis Error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
            
    def _run_monte_carlo_simulation(self, schedule_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Monte Carlo simulation for schedule risk analysis"""
        # Simulate schedule variations
        n_simulations = 1000
        simulated_durations = np.random.normal(
            loc=100,  # Baseline duration
            scale=15,  # Standard deviation
            size=n_simulations
        )
        
        # Calculate risk metrics
        risk_score = np.percentile(simulated_durations, 80) / 100  # 80th percentile
        confidence_level = np.mean(simulated_durations <= 100)  # Probability of meeting baseline
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(simulated_durations)
        
        return {
            "risk_score": risk_score,
            "confidence_level": confidence_level,
            "risk_factors": risk_factors
        }
        
    def _identify_risk_factors(self, simulated_durations: np.ndarray) -> list:
        """Identify key risk factors from simulation results"""
        risk_factors = []
        
        # Analyze simulation results for patterns
        if np.std(simulated_durations) > 20:
            risk_factors.append("High schedule variability")
        if np.mean(simulated_durations) > 110:
            risk_factors.append("Systematic schedule overrun")
        if np.percentile(simulated_durations, 95) > 130:
            risk_factors.append("Significant risk of extreme delays")
            
        return risk_factors
        
    def _assess_risk(self, risk_score: float) -> Dict[str, Any]:
        """Assess risk level based on risk score"""
        if risk_score > 1.3:
            return {
                "status": "critical",
                "risk_level": "high",
                "message": "High schedule risk detected. Immediate action required."
            }
        elif risk_score > 1.1:
            return {
                "status": "warning",
                "risk_level": "medium",
                "message": "Moderate schedule risk detected. Mitigation recommended."
            }
        else:
            return {
                "status": "good",
                "risk_level": "low",
                "message": "Schedule risk within acceptable limits."
            }
            
    def _generate_recommendations(self, risk_level: str, simulation_results: Dict[str, Any]) -> list:
        """Generate recommendations based on risk assessment"""
        recommendations = []
        
        if risk_level == "high":
            recommendations.extend([
                "Implement immediate schedule recovery plan",
                "Review and optimize critical path activities",
                "Consider additional resources for key activities",
                "Establish daily progress monitoring"
            ])
        elif risk_level == "medium":
            recommendations.extend([
                "Review buffer allocation strategy",
                "Optimize resource assignments",
                "Increase monitoring frequency",
                "Prepare contingency plans"
            ])
        else:
            recommendations.extend([
                "Continue regular schedule monitoring",
                "Update risk register periodically",
                "Maintain current control measures"
            ])
            
        # Add specific recommendations based on risk factors
        for factor in simulation_results["risk_factors"]:
            if factor == "High schedule variability":
                recommendations.append("Implement stricter schedule control measures")
            elif factor == "Systematic schedule overrun":
                recommendations.append("Review and revise duration estimates")
            elif factor == "Significant risk of extreme delays":
                recommendations.append("Develop extreme delay contingency plans")
                
        return recommendations

    def _calculate_schedule_variance(self, baseline, current):
        """Calculate schedule variance metrics"""
        try:
            baseline_duration = baseline.get('duration', 0)
            current_duration = current.get('duration', 0)
            duration_variance = current_duration - baseline_duration
            
            variance_percent = (duration_variance / baseline_duration * 100) if baseline_duration > 0 else 0
            
            return {
                'duration_variance': duration_variance,
                'variance_percent': variance_percent
            }
        except Exception as e:
            self.logger.error(f"Error calculating schedule variance: {e}")
            return None
