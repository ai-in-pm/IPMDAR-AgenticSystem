import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from PyPDF2 import PdfReader
import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import torch.optim as optim

from ..data_ingestion_agent import DataIngestionAgent
from ..evm_analyzer_agent import EVMAnalyzerAgent
from ..schedule_risk_agent import ScheduleRiskAgent
from ..compliance_agent import ComplianceAgent
from ..predictive_analytics_agent import PredictiveAnalyticsAgent
from ..report_generator_agent import ReportGeneratorAgent

class IPMDARTrainer:
    def __init__(self, guide_path: str, dataset_path: str):
        """Initialize the IPMDAR trainer"""
        self.guide_path = Path(guide_path)
        self.dataset_path = Path(dataset_path)
        self.logger = logging.getLogger(__name__)
        self.nlp = spacy.load("en_core_web_lg")
        self.sentence_transformer = SentenceTransformer('all-mpnet-base-v2')
        self.knowledge_base = {}
        
        # Initialize agents
        self.data_ingestion_agent = DataIngestionAgent()
        self.evm_analyzer_agent = EVMAnalyzerAgent()
        self.schedule_risk_agent = ScheduleRiskAgent()
        self.compliance_agent = ComplianceAgent()
        self.predictive_analytics_agent = PredictiveAnalyticsAgent()
        self.report_generator_agent = ReportGeneratorAgent()
        
        # Load training data
        with open(dataset_path, 'r') as f:
            self.training_data = json.load(f)
            
    def train_all_agents(self) -> Dict[str, Any]:
        """Train all AI agents with IPMDAR knowledge"""
        try:
            # Load and preprocess training data
            guide_content = self._extract_guide_content()
            training_data = self._load_training_dataset()
            
            # Process guide content for knowledge extraction
            self._build_knowledge_base(guide_content)
            
            # Train individual agents
            training_results = {
                "data_ingestion": self._train_data_ingestion_agent(training_data),
                "evm_analyzer": self._train_evm_analyzer_agent(training_data),
                "schedule_risk": self._train_schedule_risk_agent(training_data),
                "compliance": self._train_compliance_agent(training_data, guide_content),
                "predictive_analytics": self._train_predictive_analytics_agent(training_data),
                "report_generator": self._train_report_generator_agent(training_data, guide_content)
            }
            
            return {
                "status": "success",
                "results": training_results,
                "knowledge_base_size": len(self.knowledge_base)
            }
            
        except Exception as e:
            self.logger.error(f"Training Error: {e}")
            return {"status": "error", "error": str(e)}
            
    def _extract_guide_content(self) -> Dict[str, Any]:
        """Extract and structure content from the IPMDAR implementation guide"""
        guide_content = {
            "sections": {},
            "definitions": {},
            "requirements": [],
            "examples": []
        }
        
        try:
            reader = PdfReader(self.guide_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
                
            # Process text with spaCy for structured extraction
            doc = self.nlp(text)
            
            # Extract sections
            current_section = ""
            for sent in doc.sents:
                if sent.text.isupper() and len(sent.text.split()) <= 10:
                    current_section = sent.text.strip()
                    guide_content["sections"][current_section] = []
                elif current_section:
                    guide_content["sections"][current_section].append(sent.text)
                    
            # Extract definitions
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT", "LAW"]:
                    definition_context = text[max(0, ent.start_char - 100):min(len(text), ent.end_char + 100)]
                    if "means" in definition_context.lower() or "defined as" in definition_context.lower():
                        guide_content["definitions"][ent.text] = definition_context
                        
            # Extract requirements
            for sent in doc.sents:
                if "shall" in sent.text.lower() or "must" in sent.text.lower() or "required" in sent.text.lower():
                    guide_content["requirements"].append(sent.text)
                    
            # Extract examples
            for sent in doc.sents:
                if "example" in sent.text.lower() or "e.g." in sent.text.lower():
                    guide_content["examples"].append(sent.text)
                    
            return guide_content
            
        except Exception as e:
            self.logger.error(f"Guide Content Extraction Error: {e}")
            raise
            
    def _load_training_dataset(self) -> Dict[str, Any]:
        """Load and preprocess the IPMDAR training dataset"""
        try:
            # Extract samples from the dataset
            samples = self.training_data.get("samples", [])
            if not samples:
                raise ValueError("No training samples found in dataset")
                
            # Convert to pandas DataFrame for easier processing
            df = pd.json_normalize(samples)
            
            # Split into train/test sets
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
            
            return {
                "train": train_data,
                "test": test_data,
                "full": df
            }
            
        except Exception as e:
            self.logger.error(f"Dataset Loading Error: {e}")
            raise
            
    def _build_knowledge_base(self, guide_content: Dict[str, Any]):
        """Build a structured knowledge base from guide content"""
        # Encode all content using sentence transformer
        for section, content in guide_content["sections"].items():
            section_text = " ".join(content)
            section_embedding = self.sentence_transformer.encode(section_text)
            self.knowledge_base[section] = {
                "content": content,
                "embedding": section_embedding
            }
            
        # Add definitions to knowledge base
        for term, definition in guide_content["definitions"].items():
            term_embedding = self.sentence_transformer.encode(definition)
            self.knowledge_base[f"DEF_{term}"] = {
                "content": definition,
                "embedding": term_embedding
            }
            
        # Add requirements
        requirements_text = " ".join(guide_content["requirements"])
        req_embedding = self.sentence_transformer.encode(requirements_text)
        self.knowledge_base["REQUIREMENTS"] = {
            "content": guide_content["requirements"],
            "embedding": req_embedding
        }
            
    def _train_data_ingestion_agent(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train the Data Ingestion Agent"""
        try:
            # Train validation rules model
            validation_model = self._train_validation_model(training_data["train"])
            
            # Train data format classifier
            format_classifier = self._train_format_classifier(training_data["train"])
            
            return {
                "status": "success",
                "validation_model": validation_model,
                "format_classifier": format_classifier,
                "accuracy": self._evaluate_model(validation_model, training_data["test"])
            }
            
        except Exception as e:
            self.logger.error(f"Data Ingestion Training Error: {e}")
            return {"status": "error", "error": str(e)}
            
    def _train_evm_analyzer_agent(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train the EVM Analyzer Agent"""
        try:
            # Train performance prediction model
            performance_model = self._train_performance_model(training_data["train"])
            
            # Train threshold detection model
            threshold_model = self._train_threshold_model(training_data["train"])
            
            return {
                "status": "success",
                "performance_model": performance_model,
                "threshold_model": threshold_model,
                "accuracy": self._evaluate_model(performance_model, training_data["test"])
            }
            
        except Exception as e:
            self.logger.error(f"EVM Analyzer Training Error: {e}")
            return {"status": "error", "error": str(e)}
            
    def _train_schedule_risk_agent(self, training_data: Dict[str, Any]) -> None:
        """Train the schedule risk assessment model"""
        try:
            # Load mathematical training data
            with open('dataset/IPMDAR_Mathematical_Training.json', 'r') as f:
                math_training = json.load(f)
                
            # Extract schedule analysis problems
            schedule_problems = math_training['training_data']['schedule_analysis']
            
            # Combine standard features with mathematical features
            features = self._extract_temporal_features(training_data)
            math_features = self._extract_mathematical_features(schedule_problems)
            combined_features = pd.concat([features, math_features], axis=1)
            
            # Train the model with enhanced features
            self.schedule_risk_agent.train(combined_features, training_data['risk_levels'])
            
            # Log training results
            self.logger.info("Schedule risk agent trained successfully with mathematical enhancements")
            self.logger.info(f"Feature importance: {self.schedule_risk_agent.model.feature_importances_}")
            
        except Exception as e:
            self.logger.error(f"Error training schedule risk agent: {e}")
            raise

    def _extract_mathematical_features(self, schedule_problems: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract advanced mathematical features from schedule problems"""
        features = pd.DataFrame()
        
        for problem in schedule_problems:
            # Extract probabilistic features
            if 'uncertainty_factors' in problem['input_data'].get('baseline_schedule', {}):
                factors = problem['input_data']['baseline_schedule']['uncertainty_factors']
                features['uncertainty_impact'] = self._calculate_combined_uncertainty(factors)
                
            # Extract resource optimization features
            if 'resource_limits' in problem.get('input_data', {}):
                features['resource_constraint_factor'] = self._calculate_resource_constraints(
                    problem['input_data']['tasks'],
                    problem['input_data']['resource_limits']
                )
                
            # Extract network complexity features
            if 'tasks' in problem.get('input_data', {}):
                features['network_complexity'] = self._calculate_network_complexity(
                    problem['input_data']['tasks']
                )
        
        return features

    def _calculate_combined_uncertainty(self, uncertainty_factors: List[Dict[str, Any]]) -> float:
        """Calculate combined impact of uncertainty factors using statistical distributions"""
        combined_impact = 0
        
        for factor in uncertainty_factors:
            distribution = factor['impact_distribution']
            if 'normal' in distribution:
                mean, std = map(float, distribution.strip('normal()').split(','))
                impact = np.random.normal(mean, std)
            elif 'triangular' in distribution:
                left, mode, right = map(float, distribution.strip('triangular()').split(','))
                impact = np.random.triangular(left, mode, right)
            elif 'lognormal' in distribution:
                mu, sigma = map(float, distribution.strip('lognormal()').split(','))
                impact = np.random.lognormal(mu, sigma)
                
            combined_impact += impact
            
        return combined_impact

    def _calculate_resource_constraints(self, tasks: List[Dict[str, Any]], 
                                     resource_limits: Dict[str, int]) -> float:
        """Calculate resource constraint factor using linear programming"""
        total_demand = {resource: 0 for resource in resource_limits}
        
        for task in tasks:
            for resource, amount in task['resources'].items():
                total_demand[resource] += amount
                
        constraint_factor = max(
            total_demand[resource] / limit 
            for resource, limit in resource_limits.items()
        )
        
        return constraint_factor

    def _calculate_network_complexity(self, tasks: List[Dict[str, Any]]) -> float:
        """Calculate network complexity using graph theory metrics"""
        # Build adjacency matrix
        task_ids = [task['id'] for task in tasks]
        n = len(task_ids)
        adjacency = np.zeros((n, n))
        
        for i, task in enumerate(tasks):
            for dep in task.get('dependencies', []):
                j = task_ids.index(dep)
                adjacency[i][j] = 1
                
        # Calculate complexity metrics
        num_edges = np.sum(adjacency)
        max_possible_edges = n * (n-1) / 2
        complexity = num_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        return complexity

    def _train_compliance_agent(self, training_data: Dict[str, pd.DataFrame],
                              guide_content: Dict[str, Any]) -> Dict[str, Any]:
        """Train the Compliance Agent"""
        try:
            # Train compliance rules classifier
            compliance_model = self._train_compliance_model(training_data["train"], guide_content)
            
            # Train DCMA checks classifier
            dcma_model = self._train_dcma_model(training_data["train"])
            
            return {
                "status": "success",
                "compliance_model": compliance_model,
                "dcma_model": dcma_model,
                "accuracy": self._evaluate_model(compliance_model, training_data["test"])
            }
            
        except Exception as e:
            self.logger.error(f"Compliance Training Error: {e}")
            return {"status": "error", "error": str(e)}
            
    def _train_predictive_analytics_agent(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train the Predictive Analytics Agent"""
        try:
            # Train forecasting model
            forecast_model = self._train_forecast_model(training_data["train"])
            
            # Train confidence estimation model
            confidence_model = self._train_confidence_model(training_data["train"])
            
            return {
                "status": "success",
                "forecast_model": forecast_model,
                "confidence_model": confidence_model,
                "accuracy": self._evaluate_model(forecast_model, training_data["test"])
            }
            
        except Exception as e:
            self.logger.error(f"Predictive Analytics Training Error: {e}")
            return {"status": "error", "error": str(e)}
            
    def _train_report_generator_agent(self, training_data: Dict[str, pd.DataFrame],
                                    guide_content: Dict[str, Any]) -> Dict[str, Any]:
        """Train the Report Generator Agent"""
        try:
            # Train report template generator
            template_model = self._train_template_model(training_data["train"], guide_content)
            
            # Train recommendation prioritization model
            priority_model = self._train_priority_model(training_data["train"])
            
            return {
                "status": "success",
                "template_model": template_model,
                "priority_model": priority_model,
                "accuracy": self._evaluate_model(template_model, training_data["test"])
            }
            
        except Exception as e:
            self.logger.error(f"Report Generator Training Error: {e}")
            return {"status": "error", "error": str(e)}

    def _train_cost_agent(self, training_data: Dict[str, Any]) -> None:
        """Train the cost analysis agent with mathematical enhancements"""
        try:
            # Load mathematical training data
            with open('dataset/IPMDAR_Mathematical_Training.json', 'r') as f:
                math_training = json.load(f)
                
            # Extract cost analysis problems
            cost_problems = math_training['training_data']['cost_analysis']
            
            # Combine standard features with mathematical features
            features = self._extract_cost_features(training_data)
            math_features = self._extract_cost_mathematical_features(cost_problems)
            combined_features = pd.concat([features, math_features], axis=1)
            
            # Train the model with enhanced features
            self.cost_agent.train(combined_features, training_data['cost_variance_levels'])
            
            # Log training results
            self.logger.info("Cost agent trained successfully with mathematical enhancements")
            self.logger.info(f"Feature importance: {self.cost_agent.model.feature_importances_}")
            
        except Exception as e:
            self.logger.error(f"Error training cost agent: {e}")
            raise

    def _extract_cost_mathematical_features(self, cost_problems: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract advanced mathematical features for cost analysis"""
        features = pd.DataFrame()
        
        for problem in cost_problems:
            # Extract non-linear cost growth features
            if 'coefficients' in problem.get('input_data', {}):
                coef = problem['input_data']['coefficients']
                features['cost_growth_rate'] = self._calculate_cost_growth_rate(coef)
                
            # Extract dependency-based features
            if 'components' in problem.get('input_data', {}):
                components = problem['input_data']['components']
                features['dependency_impact'] = self._calculate_dependency_impact(components)
        
        return features

    def _calculate_cost_growth_rate(self, coefficients: Dict[str, float]) -> float:
        """Calculate cost growth rate from polynomial coefficients"""
        # For f(t) = at² + bt + c, calculate growth rate at current time
        t = coefficients.get('current_month', 6)  # Default to month 6 if not specified
        growth_rate = 2 * coefficients['a'] * t + coefficients['b']
        return growth_rate

    def _calculate_dependency_impact(self, components: List[Dict[str, Any]]) -> float:
        """Calculate cost impact based on component dependencies"""
        # Build dependency graph
        dependency_graph = {}
        for comp in components:
            dependency_graph[comp['name']] = {
                'variance': comp['actual_cost'] - comp['planned_cost'],
                'dependencies': comp.get('dependencies', [])
            }
        
        # Calculate propagated impact using topological sort
        total_impact = 0
        visited = set()
        
        def visit(component):
            if component in visited:
                return dependency_graph[component]['propagated_impact']
            
            visited.add(component)
            impact = dependency_graph[component]['variance']
            
            # Add impact from dependencies
            for dep in dependency_graph[component]['dependencies']:
                impact += 0.5 * visit(dep)  # Assume 50% impact propagation
                
            dependency_graph[component]['propagated_impact'] = impact
            return impact
        
        # Calculate total impact
        for component in dependency_graph:
            if component not in visited:
                total_impact += visit(component)
        
        return total_impact

    # Helper methods for model training
    def _train_validation_model(self, data: pd.DataFrame) -> RandomForestClassifier:
        """Train a model for data validation rules"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Implementation details...
        return model
        
    def _train_format_classifier(self, data: pd.DataFrame) -> RandomForestClassifier:
        """Train a model for data format classification"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Implementation details...
        return model
        
    def _train_performance_model(self, data: pd.DataFrame) -> RandomForestRegressor:
        """Train a model for performance prediction"""
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        # Implementation details...
        return model
        
    def _train_threshold_model(self, data: pd.DataFrame) -> RandomForestClassifier:
        """Train a model for threshold detection"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Implementation details...
        return model
        
    def _train_risk_model(self, data: pd.DataFrame) -> RandomForestClassifier:
        """Train a model for risk assessment"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Implementation details...
        return model
        
    def _train_simulation_params(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train parameters for Monte Carlo simulation"""
        params = {
            "distribution_type": "normal",
            "mean": data["duration"].mean(),
            "std": data["duration"].std()
        }
        return params
        
    def _train_compliance_model(self, data: pd.DataFrame,
                              guide_content: Dict[str, Any]) -> RandomForestClassifier:
        """Train a model for compliance checking"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Implementation details...
        return model
        
    def _train_dcma_model(self, data: pd.DataFrame) -> RandomForestClassifier:
        """Train a model for DCMA checks"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Implementation details...
        return model
        
    def _train_forecast_model(self, data: pd.DataFrame) -> RandomForestRegressor:
        """Train a model for forecasting"""
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        # Implementation details...
        return model
        
    def _train_confidence_model(self, data: pd.DataFrame) -> RandomForestRegressor:
        """Train a model for confidence estimation"""
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        # Implementation details...
        return model
        
    def _train_template_model(self, data: pd.DataFrame,
                            guide_content: Dict[str, Any]) -> RandomForestClassifier:
        """Train a model for report template generation"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Implementation details...
        return model
        
    def _train_priority_model(self, data: pd.DataFrame) -> RandomForestClassifier:
        """Train a model for recommendation prioritization"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Implementation details...
        return model
        
    def _evaluate_model(self, model: Any, test_data: pd.DataFrame) -> float:
        """Evaluate model performance"""
        # Implementation details...
        return 0.95  # Placeholder accuracy
