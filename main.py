import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import json
import time
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QPushButton, QTextEdit, QSplitter, QLabel, QComboBox,
                            QFileDialog, QHBoxLayout, QScrollArea, QFrame,
                            QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor
from agents.data_ingestion_agent import DataIngestionAgent
from agents.evm_analyzer_agent import EVMAnalyzerAgent
from agents.schedule_risk_agent import ScheduleRiskAgent
from agents.cost_agent import CostAgent
from agents.compliance_agent import ComplianceAgent
from agents.report_generator_agent import ReportGeneratorAgent

class AnomalyQuestionThread(QThread):
    question_signal = pyqtSignal(str, str)  # agent_name, question
    
    def __init__(self, anomalies: Dict[str, List[Dict]], parent=None):
        super().__init__(parent)
        self.anomalies = anomalies
        self.running = True
        
    def run(self):
        while self.running:
            for agent_name, agent_anomalies in self.anomalies.items():
                for anomaly in agent_anomalies:
                    if self.running:
                        question = self.generate_question(agent_name, anomaly)
                        self.question_signal.emit(agent_name, question)
                        time.sleep(120)  # Wait 2 minutes
                        
    def generate_question(self, agent_name: str, anomaly: Dict) -> str:
        questions = [
            f"I noticed {anomaly['description']} on line {anomaly['line_number']}. Would you like me to explain why this might be an issue?",
            f"Have you considered the impact of {anomaly['description']} on the overall project?",
            f"I can help you address the {anomaly['type']} issue found. Would you like to discuss potential solutions?",
            f"Based on my analysis, fixing {anomaly['description']} could improve project performance. Shall we look at this together?"
        ]
        return questions[int(time.time()) % len(questions)]
        
    def stop(self):
        self.running = False

class IPMDARAnalyzerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.agents = {
            "All Agents": None,
            "Data Ingestion": DataIngestionAgent(),
            "EVM Analyzer": EVMAnalyzerAgent(),
            "Schedule Risk": ScheduleRiskAgent(),
            "Cost Analysis": CostAgent(),
            "Compliance": ComplianceAgent(),
            "Report Generator": ReportGeneratorAgent()
        }
        self.current_files = []
        self.anomalies = {}
        self.question_thread = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI components"""
        self.setWindowTitle('IPMDAR AI Agent System')
        self.setGeometry(100, 100, 1400, 900)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Welcome message
        welcome_label = QLabel("Welcome to IPMDAR AI Agent System")
        welcome_label.setFont(QFont('Arial', 14, QFont.Bold))
        welcome_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(welcome_label)
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        # Agent selection
        agent_layout = QVBoxLayout()
        agent_label = QLabel("Select AI Agent:")
        self.agent_combo = QComboBox()
        self.agent_combo.addItems(self.agents.keys())
        agent_layout.addWidget(agent_label)
        agent_layout.addWidget(self.agent_combo)
        controls_layout.addLayout(agent_layout)
        
        # File controls
        file_buttons_layout = QVBoxLayout()
        self.import_button = QPushButton('Import JSON Files')
        self.import_button.clicked.connect(self.import_files)
        self.export_button = QPushButton('Export Report')
        self.export_button.clicked.connect(self.export_report)
        file_buttons_layout.addWidget(self.import_button)
        file_buttons_layout.addWidget(self.export_button)
        controls_layout.addLayout(file_buttons_layout)
        
        layout.addLayout(controls_layout)
        
        # Create main content splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Chat interface
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        chat_label = QLabel("AI Agent Chat")
        self.chat_output = QTextEdit()
        self.chat_output.setReadOnly(True)
        chat_layout.addWidget(chat_label)
        chat_layout.addWidget(self.chat_output)
        
        # Analysis output
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        analysis_label = QLabel("Analysis Results")
        self.analysis_output = QTextEdit()
        self.analysis_output.setReadOnly(True)
        analysis_layout.addWidget(analysis_label)
        analysis_layout.addWidget(self.analysis_output)
        
        # Add widgets to splitter
        splitter.addWidget(chat_widget)
        splitter.addWidget(analysis_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Initialize welcome message
        self.display_welcome_message()
        
    def display_welcome_message(self):
        """Display the welcome message in the chat interface"""
        welcome_text = """Welcome to the IPMDAR AI Agent System! 

I'm here to help you analyze your IPMDAR data with our team of specialized AI agents:
- Data Ingestion Agent: Validates and processes your input data
- EVM Analyzer Agent: Performs earned value management analysis
- Schedule Risk Agent: Assesses schedule risks using advanced mathematical models
- Cost Agent: Analyzes cost patterns and dependencies
- Compliance Agent: Ensures adherence to IPMDAR standards
- Report Generator Agent: Creates comprehensive analysis reports

To get started:
1. Use the dropdown menu to select an agent (or "All Agents")
2. Click "Import JSON Files" to load your IPMDAR data
3. I'll analyze your data and provide real-time insights

How can I assist you today?"""
        
        self.chat_output.setText(welcome_text)
        
    def import_files(self):
        """Import JSON files for analysis"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select JSON Files",
            "",
            "JSON Files (*.json)"
        )
        
        if files:
            self.current_files = files
            self.progress_bar.setMaximum(len(files) * 100)
            self.progress_bar.setValue(0)
            self.analyze_files()
            
    def analyze_files(self):
        """Analyze the imported JSON files"""
        self.anomalies = {}
        self.analysis_output.clear()
        
        selected_agent = self.agent_combo.currentText()
        agents_to_use = list(self.agents.values())[1:] if selected_agent == "All Agents" else [self.agents[selected_agent]]
        
        for file_path in self.current_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                file_name = os.path.basename(file_path)
                self.analysis_output.append(f"\n=== Analyzing {file_name} ===\n")
                
                for agent in agents_to_use:
                    agent_name = agent.__class__.__name__
                    analysis_result = agent.analyze(data)
                    
                    if 'anomalies' in analysis_result:
                        self.anomalies[agent_name] = analysis_result['anomalies']
                        self.display_anomalies(agent_name, analysis_result['anomalies'])
                        
            except Exception as e:
                self.analysis_output.append(f"Error analyzing {file_path}: {str(e)}\n")
                
        # Start anomaly question thread
        if self.anomalies:
            if self.question_thread and self.question_thread.isRunning():
                self.question_thread.stop()
                self.question_thread.wait()
                
            self.question_thread = AnomalyQuestionThread(self.anomalies)
            self.question_thread.question_signal.connect(self.display_question)
            self.question_thread.start()
            
    def display_anomalies(self, agent_name: str, anomalies: List[Dict]):
        """Display anomalies found by an agent"""
        self.analysis_output.append(f"\n{agent_name} Analysis Results:")
        
        if not anomalies:
            self.analysis_output.append("No anomalies found.\n")
            return
            
        for anomaly in anomalies:
            self.analysis_output.append(f"\nAnomaly Type: {anomaly['type']}")
            self.analysis_output.append(f"Location: Line {anomaly['line_number']}")
            self.analysis_output.append(f"Description: {anomaly['description']}")
            self.analysis_output.append(f"Recommendation: {anomaly['recommendation']}\n")
            
    def display_question(self, agent_name: str, question: str):
        """Display a question in the chat interface"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_output.append(f"\n[{timestamp}] {agent_name}:\n{question}\n")
        cursor = self.chat_output.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_output.setTextCursor(cursor)
        
    def export_report(self):
        """Export analysis results to a file"""
        if not self.current_files:
            QMessageBox.warning(self, "Warning", "No analysis results to export.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Report",
            "",
            "JSON Files (*.json);;Text Files (*.txt)"
        )
        
        if file_path:
            try:
                report = {
                    "timestamp": datetime.now().isoformat(),
                    "analyzed_files": self.current_files,
                    "anomalies": self.anomalies
                }
                
                with open(file_path, 'w') as f:
                    if file_path.endswith('.json'):
                        json.dump(report, f, indent=2)
                    else:
                        f.write(self.analysis_output.toPlainText())
                        
                QMessageBox.information(self, "Success", "Report exported successfully!")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export report: {str(e)}")
                
    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        if self.question_thread and self.question_thread.isRunning():
            self.question_thread.stop()
            self.question_thread.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look and feel
    window = IPMDARAnalyzerUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
