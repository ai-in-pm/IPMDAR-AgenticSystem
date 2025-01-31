# IPMDAR AI Agent System

An advanced AI system for analyzing and processing Integrated Program Management Data Analysis Reports (IPMDAR) with PhD-level understanding and advanced mathematical capabilities.

![image](https://github.com/user-attachments/assets/584f1c7c-3f7c-41a5-a5ba-817b5222d5cc)


## Features

- **Data Ingestion**: Automated parsing and validation of IPMDAR data
- **EVM Analysis**: Advanced Earned Value Management analysis with non-linear cost modeling
- **Schedule Risk Assessment**: ML-based schedule risk prediction with advanced mathematical features:
  - Resource-constrained critical path analysis
  - Multi-factor uncertainty modeling
  - Network complexity metrics
  - Monte Carlo simulation with correlated factors
- **Cost Analysis**: Enhanced cost prediction with:
  - Non-linear cost growth modeling
  - Component dependency analysis
  - Graph-based variance propagation
  - Automated anomaly detection
  - Real-time variance tracking
- **Compliance Checking**: Automated compliance verification against IPMDAR standards
- **Predictive Analytics**: Future performance prediction using historical data and mathematical models
- **Report Generation**: Automated generation of analysis reports with confidence intervals

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Install PyQt5 for the user interface:
```bash
pip install PyQt5
```

## Running the Application

There are two ways to run the IPMDAR Analyzer:

1. Using the batch file:
```bash
run_analyzer.bat
```

2. Using Python directly:
```bash
python main.py
```

## Using the Application

1. **Launch**: Start the application using one of the methods above
2. **Select Agent**: Use the dropdown menu to select a specific agent or "All Agents"
3. **Import Data**: Click "Import Files" to select JSON files for analysis
4. **View Results**: 
   - Anomalies will be displayed in the main window
   - Agents will ask questions about detected issues
   - Detailed metrics are calculated for each analysis
5. **Export**: Use "Export Report" to save analysis results

## Project Structure

```
IPMDAR-AgenticSystem/
├── agents/                 # AI agent implementations
│   ├── base_agent.py      # Abstract base class for all agents
│   ├── data_ingestion_agent.py
│   ├── evm_analyzer_agent.py
│   ├── schedule_risk_agent.py
│   ├── cost_agent.py
│   ├── compliance_agent.py
│   ├── predictive_analytics_agent.py
│   └── report_generator_agent.py
├── dataset/               # Training data and implementation guide
│   ├── IPMDAR_Complete_Detailed_Training_Dataset.json
│   └── IPMDAR_Mathematical_Training.json    # Advanced mathematical training data
├── training/             # Training modules and utilities
├── main.py              # Main application entry point
├── run_analyzer.bat     # Batch file to run the application
└── README.md
```

## Advanced Analysis Capabilities

### Schedule Risk Assessment

The schedule risk assessment module uses advanced mathematical and machine learning techniques:

- **Temporal Analysis**:
  - Activity duration and float analysis
  - Critical path impact assessment
  - Historical performance correlation

- **Resource Optimization**:
  - Linear programming for resource allocation
  - Resource constraint impact analysis
  - Multi-resource scheduling optimization

- **Uncertainty Modeling**:
  - Statistical distribution fitting
  - Correlated Monte Carlo simulation
  - Confidence interval calculation

- **Network Analysis**:
  - Graph theory-based complexity metrics
  - Dependency chain analysis
  - Topological sorting for impact propagation

### Cost Analysis

The cost analysis module incorporates advanced mathematical modeling:

- **Non-linear Cost Growth**:
  - Polynomial regression modeling
  - Growth rate analysis
  - Trend forecasting

- **Component Dependencies**:
  - Graph-based dependency modeling
  - Variance propagation analysis
  - Automated anomaly detection
  - Real-time cost variance tracking
  - Threshold-based alerting

### Interactive Features

The system now includes an interactive user interface with:

- **Real-time Analysis**: Immediate feedback on imported data
- **Agent Communication**: Direct interaction with AI agents through chat interface
- **Anomaly Questions**: Agents proactively ask about detected issues
- **Progress Tracking**: Visual progress bar for analysis status
- **Export Capabilities**: Save analysis results and recommendations

## Environment Variables

The system uses the following API keys (optional):
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GROQ_API_KEY
- GOOGLE_API_KEY
- COHERE_API_KEY
- EMERGENCEAI_API_KEY

Store these in a `.env` file in the root directory.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - See LICENSE file for details
