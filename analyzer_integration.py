import subprocess
import os
import sys
import json
from typing import Dict, Any
from agents.agent_orchestrator import AgentOrchestrator
from pathlib import Path

class AnalyzerIntegration:
    def __init__(self, user_role: str = "analyst"):
        self.analyzer_path = Path(__file__).parent / "analyzer_application.exe"
        self.agent_orchestrator = AgentOrchestrator(user_role)
        self.user_role = user_role
        
    def set_user_role(self, role: str):
        """Update the user role for the integration"""
        self.user_role = role
        self.agent_orchestrator = AgentOrchestrator(role)
        
    def launch_analyzer(self) -> subprocess.Popen:
        """Launch the analyzer application as a subprocess"""
        try:
            process = subprocess.Popen(
                [str(self.analyzer_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            return process
        except Exception as e:
            print(f"Error launching analyzer: {e}")
            return None
            
    def process_ipmdar_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process IPMDAR data through AI agents"""
        return self.agent_orchestrator.process_ipmdar_data(data)
        
    def enhance_analyzer_output(self, analyzer_output: Dict[str, Any], user_role: str = None) -> Dict[str, Any]:
        """Enhance analyzer output with AI agent insights"""
        if user_role:
            self.set_user_role(user_role)
            
        try:
            # Process the analyzer output through our AI agents
            enhanced_results = self.process_ipmdar_data(analyzer_output)
            
            # Combine original analyzer output with AI insights
            enhanced_output = {
                "status": enhanced_results.get("status", "unknown"),
                "original_analysis": analyzer_output,
                "ai_enhanced_analysis": enhanced_results,
                "recommendations": enhanced_results.get("recommendations", []),
                "user_role": self.user_role
            }
            
            return enhanced_output
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "user_role": self.user_role
            }

def main():
    integration = AnalyzerIntegration()
    
    # Launch the analyzer application
    analyzer_process = integration.launch_analyzer()
    if not analyzer_process:
        sys.exit(1)
        
    try:
        while True:
            # Read output from the analyzer
            output = analyzer_process.stdout.readline()
            if output == '' and analyzer_process.poll() is not None:
                break
                
            if output:
                try:
                    # Try to parse output as JSON
                    analyzer_data = json.loads(output.strip())
                    
                    # Process through AI agents
                    enhanced_results = integration.enhance_analyzer_output(analyzer_data)
                    
                    # Print enhanced results
                    print(json.dumps(enhanced_results, indent=2))
                except json.JSONDecodeError:
                    # If not JSON, print as is
                    print(output.strip())
                    
    except KeyboardInterrupt:
        print("\nShutting down...")
        analyzer_process.terminate()
        
    finally:
        # Ensure proper cleanup
        analyzer_process.wait()
        
if __name__ == "__main__":
    main()
