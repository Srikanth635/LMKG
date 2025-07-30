"""
Configuration settings for SOMA Object Analysis system
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the SOMA Object Analysis system"""

    # API Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-4")
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2000"))

    # System Configuration
    DEFAULT_OUTPUT_DIR: str = os.getenv("DEFAULT_OUTPUT_DIR", "soma_analysis_results")
    DEFAULT_CONFIDENCE_THRESHOLD: float = float(os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", "0.5"))
    ENABLE_VALIDATION: bool = os.getenv("ENABLE_VALIDATION", "true").lower() == "true"

    # Knowledge Graph Configuration
    KG_EXPORT_FORMATS: list = ["turtle", "json-ld", "graphml"]
    DEFAULT_NAMESPACE_PREFIX: str = "http://example.org/instances#"

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # SOMA Ontology Configuration
    SOMA_NAMESPACE: str = "http://www.ease-crc.org/ont/SOMA.owl#"
    DUL_NAMESPACE: str = "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#"

    @classmethod
    def get_namespaces(cls) -> Dict[str, str]:
        """Get standard SOMA namespaces"""
        return {
            'soma': cls.SOMA_NAMESPACE,
            'dul': cls.DUL_NAMESPACE,
            'instance': cls.DEFAULT_NAMESPACE_PREFIX,
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
            'xsd': 'http://www.w3.org/2001/XMLSchema#'
        }

    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        if not cls.OPENAI_API_KEY:
            print("⚠️  Warning: OPENAI_API_KEY not set. LLM features will be disabled.")
            return False
        return True

    @classmethod
    def get_output_dir(cls, custom_dir: Optional[str] = None) -> Path:
        """Get output directory as Path object"""
        output_dir = custom_dir or cls.DEFAULT_OUTPUT_DIR
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


# Global configuration instance
config = Config()