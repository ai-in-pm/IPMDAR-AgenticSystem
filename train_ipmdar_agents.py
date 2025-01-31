import logging
import os
from pathlib import Path
from agents.training.ipmdar_trainer import IPMDARTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ipmdar_training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    try:
        # Define paths
        base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        guide_path = base_dir / "dataset" / "IPMDAR Implementation Guide - Oct2020 - For Public Release - Signature Edition.pdf"
        dataset_path = base_dir / "dataset" / "IPMDAR_Complete_Detailed_Training_Dataset.json"
        
        logger.info("Starting IPMDAR AI Agents specialized training...")
        logger.info(f"Using guide: {guide_path}")
        logger.info(f"Using dataset: {dataset_path}")
        
        # Initialize trainer
        trainer = IPMDARTrainer(str(guide_path), str(dataset_path))
        
        # Train all agents
        results = trainer.train_all_agents()
        
        if results["status"] == "success":
            logger.info("IPMDAR specialized training completed successfully!")
            logger.info(f"Knowledge base size: {results['knowledge_base_size']} entries")
            
            # Log individual agent results
            for agent, result in results["results"].items():
                if result["status"] == "success":
                    logger.info(f"{agent} training completed - Accuracy: {result.get('accuracy', 'N/A'):.2%}")
                else:
                    logger.error(f"{agent} training failed: {result.get('error', 'Unknown error')}")
        else:
            logger.error(f"Training failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Training process error: {e}")
        raise

if __name__ == "__main__":
    main()
