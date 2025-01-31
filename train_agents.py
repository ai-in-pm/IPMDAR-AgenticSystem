import logging
from pathlib import Path
from agents.training.ipmdar_trainer import IPMDARTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    try:
        # Define paths
        guide_path = Path("dataset/IPMDAR Implementation Guide - Oct2020 - For Public Release - Signature Edition.pdf")
        dataset_path = Path("dataset/IPMDAR_Complete_Detailed_Training_Dataset.json")
        
        logger.info("Starting IPMDAR AI Agents training...")
        
        # Initialize trainer
        trainer = IPMDARTrainer(guide_path, dataset_path)
        
        # Train all agents
        results = trainer.train_all_agents()
        
        if results["status"] == "success":
            logger.info("Training completed successfully!")
            logger.info(f"Knowledge base size: {results['knowledge_base_size']} entries")
            
            # Log individual agent results
            for agent, result in results["results"].items():
                logger.info(f"{agent} training accuracy: {result.get('accuracy', 'N/A')}")
        else:
            logger.error(f"Training failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Training process error: {e}")
        raise

if __name__ == "__main__":
    main()
