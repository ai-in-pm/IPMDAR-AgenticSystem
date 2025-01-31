from typing import Dict, Any, List
from datetime import datetime
from .base_agent import BaseAgent

class ReportGeneratorAgent(BaseAgent):
    def __init__(self, user_role: str = "analyst"):
        super().__init__(user_role)
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates comprehensive project performance report
        
        Args:
            data: Dictionary containing analysis results from all agents
            
        Returns:
            Dict containing formatted report
        """
        if not self.check_access("report"):
            return {"error": "Unauthorized access"}
            
        problem = "Generating comprehensive project performance report"
        process_steps = [
            "Step 1: Checking user authorization",
            "Step 2: Aggregating agent insights",
            "Step 3: Identifying key findings",
            "Step 4: Prioritizing recommendations",
            "Step 5: Formatting report sections"
        ]
        
        try:
            # Extract insights from each agent's analysis
            key_findings = self._extract_key_findings(data)
            prioritized_recommendations = self._prioritize_recommendations(data)
            
            # Generate report sections
            executive_summary = self._generate_executive_summary(key_findings)
            detailed_analysis = self._generate_detailed_analysis(data)
            action_items = self._generate_action_items(prioritized_recommendations)
            
            justification = f"Report compilation based on {len(key_findings)} key findings " \
                           f"and {len(prioritized_recommendations)} recommendations"
                           
            output = {
                "status": self._determine_overall_status(key_findings),
                "timestamp": datetime.now().isoformat(),
                "executive_summary": executive_summary,
                "key_findings": key_findings,
                "detailed_analysis": detailed_analysis,
                "recommendations": prioritized_recommendations,
                "action_items": action_items
            }
            
            # Record Chain of Thought
            cot_record = self.chain_of_thought(problem, process_steps, justification, output)
            output["reasoning"] = cot_record
            
            return output
            
        except Exception as e:
            self.logger.error(f"Report Generation Error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
            
    def _extract_key_findings(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key findings from agent analyses"""
        findings = []
        
        # Extract EVM findings
        if "evm_analysis" in data:
            evm = data["evm_analysis"]
            findings.append({
                "category": "Cost & Schedule Performance",
                "status": evm.get("status", "unknown"),
                "finding": evm.get("message", "No EVM data available"),
                "metrics": evm.get("metrics", {})
            })
            
        # Extract schedule risk findings
        if "schedule_analysis" in data:
            schedule = data["schedule_analysis"]
            findings.append({
                "category": "Schedule Risk",
                "status": schedule.get("status", "unknown"),
                "finding": schedule.get("message", "No schedule risk data available"),
                "risk_level": schedule.get("risk_level", "unknown")
            })
            
        # Extract compliance findings
        if "compliance_check" in data:
            compliance = data["compliance_check"]
            findings.append({
                "category": "Compliance",
                "status": compliance.get("status", "unknown"),
                "finding": compliance.get("message", "No compliance data available"),
                "issues": len(compliance.get("missing_fields", [])) + \
                         len(compliance.get("integrity_issues", [])) + \
                         len(compliance.get("dcma_assessment", {}).get("failed_checks", []))
            })
            
        # Extract predictive findings
        if "predictions" in data:
            predictions = data["predictions"]
            findings.append({
                "category": "Forecasting",
                "status": predictions.get("status", "unknown"),
                "finding": predictions.get("message", "No forecast data available"),
                "confidence": predictions.get("completion_estimate", {}).get("confidence_level", 0)
            })
            
        return findings
        
    def _prioritize_recommendations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize recommendations from all agents"""
        all_recommendations = []
        priority_map = {"critical": 1, "warning": 2, "good": 3}
        
        def add_recommendations(source: str, recommendations: List[str], status: str):
            for rec in recommendations:
                all_recommendations.append({
                    "source": source,
                    "recommendation": rec,
                    "priority": priority_map.get(status, 4),
                    "status": status
                })
                
        # Collect recommendations from each agent
        if "evm_analysis" in data:
            add_recommendations(
                "EVM Analysis",
                data["evm_analysis"].get("recommendations", []),
                data["evm_analysis"].get("status", "unknown")
            )
            
        if "schedule_analysis" in data:
            add_recommendations(
                "Schedule Risk",
                data["schedule_analysis"].get("recommendations", []),
                data["schedule_analysis"].get("status", "unknown")
            )
            
        if "compliance_check" in data:
            add_recommendations(
                "Compliance",
                data["compliance_check"].get("recommendations", []),
                data["compliance_check"].get("status", "unknown")
            )
            
        if "predictions" in data:
            add_recommendations(
                "Predictive Analytics",
                data["predictions"].get("recommendations", []),
                data["predictions"].get("status", "unknown")
            )
            
        # Sort by priority
        return sorted(all_recommendations, key=lambda x: x["priority"])
        
    def _generate_executive_summary(self, key_findings: List[Dict[str, Any]]) -> str:
        """Generate executive summary from key findings"""
        critical_findings = [f for f in key_findings if f["status"] == "critical"]
        warning_findings = [f for f in key_findings if f["status"] == "warning"]
        
        summary_parts = []
        
        if critical_findings:
            summary_parts.append("CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION:")
            for finding in critical_findings:
                summary_parts.append(f"- {finding['category']}: {finding['finding']}")
                
        if warning_findings:
            summary_parts.append("\nWARNING ITEMS REQUIRING MONITORING:")
            for finding in warning_findings:
                summary_parts.append(f"- {finding['category']}: {finding['finding']}")
                
        if not critical_findings and not warning_findings:
            summary_parts.append("Project is performing within acceptable parameters.")
            
        return "\n".join(summary_parts)
        
    def _generate_detailed_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed analysis section"""
        return {
            "cost_performance": self._format_cost_analysis(data.get("evm_analysis", {})),
            "schedule_performance": self._format_schedule_analysis(
                data.get("schedule_analysis", {}),
                data.get("predictions", {})
            ),
            "compliance_status": self._format_compliance_analysis(data.get("compliance_check", {}))
        }
        
    def _format_cost_analysis(self, evm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format cost performance analysis"""
        return {
            "status": evm_data.get("status", "unknown"),
            "metrics": evm_data.get("metrics", {}),
            "trend": evm_data.get("trend", "stable"),
            "analysis": evm_data.get("message", "No cost analysis available")
        }
        
    def _format_schedule_analysis(self, schedule_data: Dict[str, Any],
                                prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format schedule performance analysis"""
        return {
            "current_status": schedule_data.get("status", "unknown"),
            "risk_level": schedule_data.get("risk_level", "unknown"),
            "forecast": {
                "completion_estimate": prediction_data.get("completion_estimate", {}),
                "confidence_level": prediction_data.get("confidence_level", 0)
            }
        }
        
    def _format_compliance_analysis(self, compliance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format compliance analysis"""
        return {
            "status": compliance_data.get("status", "unknown"),
            "missing_fields": compliance_data.get("missing_fields", []),
            "integrity_issues": compliance_data.get("integrity_issues", []),
            "dcma_assessment": compliance_data.get("dcma_assessment", {})
        }
        
    def _generate_action_items(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate prioritized action items from recommendations"""
        action_items = []
        
        for i, rec in enumerate(recommendations, 1):
            action_items.append({
                "id": i,
                "priority": rec["priority"],
                "source": rec["source"],
                "action": rec["recommendation"],
                "status": "pending",
                "due_date": self._suggest_due_date(rec["priority"])
            })
            
        return action_items
        
    def _suggest_due_date(self, priority: int) -> str:
        """Suggest due date based on priority"""
        today = datetime.now()
        if priority == 1:  # Critical
            due_date = today.replace(day=today.day + 1)
        elif priority == 2:  # Warning
            due_date = today.replace(day=today.day + 7)
        else:  # Normal
            due_date = today.replace(day=today.day + 14)
        return due_date.strftime("%Y-%m-%d")
        
    def _determine_overall_status(self, findings: List[Dict[str, Any]]) -> str:
        """Determine overall project status"""
        if any(f["status"] == "critical" for f in findings):
            return "critical"
        elif any(f["status"] == "warning" for f in findings):
            return "warning"
        return "good"
