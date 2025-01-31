from typing import Dict, Any
import pandas as pd
import json
import xml.etree.ElementTree as ET
from .base_agent import BaseAgent

class DataIngestionAgent(BaseAgent):
    def __init__(self, user_role: str = "analyst"):
        super().__init__(user_role)
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and validate incoming IPMDAR data
        
        Args:
            data: Raw IPMDAR data in various formats
            
        Returns:
            Dict containing processed and validated data
        """
        if not self.check_access("ingest"):
            return {"error": "Unauthorized access"}
            
        # Chain of Thought Documentation
        problem = "Validating and preprocessing incoming IPMDAR data"
        process_steps = [
            "Step 1: Checking user authorization",
            "Step 2: Detecting data format",
            "Step 3: Validating data structure and completeness",
            "Step 4: Cleaning and normalizing data"
        ]
        
        try:
            # Parse and validate data
            parsed_data = self._parse_data(data)
            validation_results = self._validate_data(parsed_data)
            
            if validation_results["missing_fields"]:
                justification = f"Data validation failed: Missing required fields: {validation_results['missing_fields']}"
                output = {
                    "status": "error",
                    "validation_results": validation_results
                }
            else:
                # Clean and normalize data
                cleaned_data = self._clean_data(parsed_data)
                justification = "Data has been successfully parsed, validated, and cleaned according to IPMDAR standards"
                output = {
                    "status": "success",
                    "processed_data": cleaned_data,
                    "validation_results": validation_results
                }
            
            # Record Chain of Thought
            cot_record = self.chain_of_thought(problem, process_steps, justification, output)
            output["reasoning"] = cot_record
            
            return output
            
        except Exception as e:
            self.logger.error(f"Data Ingestion Error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _parse_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse data based on format"""
        if isinstance(data, str):
            try:
                # Try JSON first
                return json.loads(data)
            except json.JSONDecodeError:
                try:
                    # Try XML
                    return self._parse_xml(data)
                except ET.ParseError:
                    raise ValueError("Unsupported data format")
        return data
    
    def _validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data structure and completeness"""
        validation_results = {
            "missing_fields": [],
            "invalid_formats": [],
            "warnings": []
        }
        
        required_fields = [
            "IPMDAR_Records",
            "CPI",
            "SPI"
        ]
        
        for field in required_fields:
            if field not in data:
                validation_results["missing_fields"].append(field)
        
        return validation_results
    
    def _clean_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize data"""
        cleaned_data = {}
        
        for key, value in data.items():
            # Remove null values
            if value is not None:
                # Convert to appropriate data type
                if isinstance(value, (int, float, bool, str)):
                    cleaned_data[key] = value
                elif isinstance(value, dict):
                    cleaned_data[key] = self._clean_data(value)
                elif isinstance(value, list):
                    cleaned_data[key] = [
                        self._clean_data(item) if isinstance(item, dict) else item
                        for item in value
                    ]
        
        return cleaned_data
    
    def _parse_xml(self, xml_string: str) -> Dict[str, Any]:
        """Parse XML data into dictionary"""
        root = ET.fromstring(xml_string)
        return self._xml_to_dict(root)
    
    def _xml_to_dict(self, element) -> Dict[str, Any]:
        """Convert XML element to dictionary"""
        result = {}
        
        for child in element:
            if len(child) == 0:
                result[child.tag] = child.text
            else:
                result[child.tag] = self._xml_to_dict(child)
                
        return result
