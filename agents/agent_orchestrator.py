from typing import Dict, List
import logging
from .data_ingestion_agent import DataIngestionAgent
from .evm_analyzer_agent import EVMAnalyzerAgent
from .schedule_risk_agent import ScheduleRiskAgent
from .compliance_agent import ComplianceAgent
from .predictive_analytics_agent import PredictiveAnalyticsAgent
from .report_generator_agent import ReportGeneratorAgent

class AgentOrchestrator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.agents = {
            'data_ingestion': DataIngestionAgent(),
            'evm_analyzer': EVMAnalyzerAgent(),
            'schedule_risk': ScheduleRiskAgent(),
            'compliance': ComplianceAgent(),
            'predictive_analytics': PredictiveAnalyticsAgent(),
            'report_generator': ReportGeneratorAgent()
        }
        
    def process_ipmdar_data(self, data: Dict) -> Dict:
        """
        Orchestrate the processing of IPMDAR data through all agents
        
        Args:
            data (Dict): Raw IPMDAR data
            
        Returns:
            Dict: Processed results and recommendations
        """
        results = {}
        
        # Step 1: Data Ingestion
        processed_data = self.agents['data_ingestion'].process(data)
        results['data_validation'] = processed_data
        
        # Step 2: Parallel Processing
        # EVM Analysis
        evm_results = self.agents['evm_analyzer'].process(processed_data)
        results['evm_analysis'] = evm_results
        
        # Schedule Risk Analysis
        schedule_results = self.agents['schedule_risk'].process(processed_data)
        results['schedule_analysis'] = schedule_results
        
        # Compliance Check
        compliance_results = self.agents['compliance'].process(processed_data)
        results['compliance_check'] = compliance_results
        
        # Step 3: Predictive Analytics
        predictive_results = self.agents['predictive_analytics'].process({
            'evm_data': evm_results,
            'schedule_data': schedule_results,
            'compliance_data': compliance_results
        })
        results['predictions'] = predictive_results
        
        # Step 4: Generate Report
        final_report = self.agents['report_generator'].process(results)
        results['executive_summary'] = final_report
        
        return results
    
    def get_agent_status(self) -> List[Dict]:
        """Get the status of all agents"""
        return [{name: agent.__class__.__name__} for name, agent in self.agents.items()]
