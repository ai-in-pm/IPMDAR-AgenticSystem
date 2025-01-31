from typing import Dict, Any, List
import hashlib
from datetime import datetime
from .base_agent import BaseAgent

class ComplianceAgent(BaseAgent):
    def __init__(self, user_role: str = "analyst"):
        super().__init__(user_role)
        self.required_fields = {
            "IPMDAR_Records": [
                "contract_data",
                "performance_data",
                "schedule_data"
            ],
            "performance_data": [
                "CPI",
                "SPI",
                "BCWS",
                "BCWP",
                "ACWP"
            ],
            "schedule_data": [
                "baseline_schedule",
                "current_schedule",
                "critical_path",
                "float_analysis"
            ]
        }
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates IPMDAR compliance with reporting standards
        
        Args:
            data: IPMDAR data to validate
            
        Returns:
            Dict containing compliance analysis results
        """
        if not self.check_access("analyze"):
            return {"error": "Unauthorized access"}
            
        problem = "Validating IPMDAR data compliance and integrity"
        process_steps = [
            "Step 1: Checking user authorization",
            "Step 2: Validating required fields",
            "Step 3: Checking data integrity",
            "Step 4: Validating DCMA 14-point assessment",
            "Step 5: Generating compliance report"
        ]
        
        try:
            # Perform compliance checks
            missing_fields = self._check_required_fields(data)
            integrity_check = self._check_data_integrity(data)
            dcma_assessment = self._perform_dcma_assessment(data)
            
            # Determine compliance status
            compliance_status = self._determine_compliance_status(
                missing_fields,
                integrity_check["issues"],
                dcma_assessment["failed_checks"]
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                missing_fields,
                integrity_check["issues"],
                dcma_assessment["failed_checks"]
            )
            
            justification = f"Compliance assessment based on {len(missing_fields)} missing fields, " \
                           f"{len(integrity_check['issues'])} integrity issues, and " \
                           f"{len(dcma_assessment['failed_checks'])} failed DCMA checks"
                           
            output = {
                "status": compliance_status["status"],
                "message": compliance_status["message"],
                "missing_fields": missing_fields,
                "integrity_issues": integrity_check["issues"],
                "dcma_assessment": dcma_assessment,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat(),
                "data_hash": integrity_check["hash"]
            }
            
            # Record Chain of Thought
            cot_record = self.chain_of_thought(problem, process_steps, justification, output)
            output["reasoning"] = cot_record
            
            return output
            
        except Exception as e:
            self.logger.error(f"Compliance Analysis Error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
            
    def _check_required_fields(self, data: Dict[str, Any]) -> List[str]:
        """Check for required fields in the IPMDAR data"""
        missing_fields = []
        
        def check_nested_fields(data_dict: Dict[str, Any], required: Dict[str, List[str]], prefix: str = ""):
            for field, subfields in required.items():
                full_field = f"{prefix}{field}" if prefix else field
                if field not in data_dict:
                    missing_fields.append(full_field)
                elif isinstance(subfields, list):
                    for subfield in subfields:
                        if subfield not in data_dict[field]:
                            missing_fields.append(f"{full_field}.{subfield}")
                            
        check_nested_fields(data, self.required_fields)
        return missing_fields
        
    def _check_data_integrity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check data integrity and generate hash"""
        issues = []
        
        # Generate data hash
        data_str = str(sorted(data.items()))
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()
        
        # Check for data consistency
        if "performance_data" in data:
            perf_data = data["performance_data"]
            if all(k in perf_data for k in ["BCWS", "BCWP", "ACWP", "CPI", "SPI"]):
                # Verify CPI calculation
                calculated_cpi = perf_data["BCWP"] / perf_data["ACWP"] if perf_data["ACWP"] != 0 else 0
                if abs(calculated_cpi - perf_data["CPI"]) > 0.01:
                    issues.append("CPI calculation inconsistency detected")
                    
                # Verify SPI calculation
                calculated_spi = perf_data["BCWP"] / perf_data["BCWS"] if perf_data["BCWS"] != 0 else 0
                if abs(calculated_spi - perf_data["SPI"]) > 0.01:
                    issues.append("SPI calculation inconsistency detected")
                    
        return {
            "hash": data_hash,
            "issues": issues
        }
        
    def _perform_dcma_assessment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform DCMA 14-point schedule assessment"""
        failed_checks = []
        
        if "schedule_data" in data:
            schedule = data["schedule_data"]
            
            # Check 1: Invalid dates
            if "baseline_schedule" not in schedule or "current_schedule" not in schedule:
                failed_checks.append("Missing schedule data")
                
            # Check 2: Critical path analysis
            if "critical_path" not in schedule:
                failed_checks.append("Missing critical path analysis")
                
            # Check 3: Float analysis
            if "float_analysis" not in schedule:
                failed_checks.append("Missing float analysis")
                
        return {
            "failed_checks": failed_checks,
            "pass_rate": 1 - (len(failed_checks) / 14)  # Simplified for example
        }
        
    def _determine_compliance_status(self, missing_fields: List[str],
                                   integrity_issues: List[str],
                                   failed_checks: List[str]) -> Dict[str, str]:
        """Determine overall compliance status"""
        total_issues = len(missing_fields) + len(integrity_issues) + len(failed_checks)
        
        if total_issues == 0:
            return {
                "status": "compliant",
                "message": "Data fully compliant with IPMDAR standards"
            }
        elif total_issues < 3:
            return {
                "status": "warning",
                "message": "Minor compliance issues detected"
            }
        else:
            return {
                "status": "non_compliant",
                "message": "Significant compliance issues detected"
            }
            
    def _generate_recommendations(self, missing_fields: List[str],
                                integrity_issues: List[str],
                                failed_checks: List[str]) -> List[str]:
        """Generate recommendations based on compliance issues"""
        recommendations = []
        
        if missing_fields:
            recommendations.extend([
                f"Add missing field: {field}" for field in missing_fields
            ])
            
        if integrity_issues:
            recommendations.extend([
                f"Resolve data integrity issue: {issue}" for issue in integrity_issues
            ])
            
        if failed_checks:
            recommendations.extend([
                f"Address DCMA check failure: {check}" for check in failed_checks
            ])
            
        if not recommendations:
            recommendations.append("Maintain current compliance standards")
            
        return recommendations
