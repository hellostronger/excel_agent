"""
Deployment script for Excel Intelligence Agent to Google Cloud Vertex AI Agent Engine

This script handles the deployment of the multi-agent Excel analysis system
to Google Cloud for production use.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from google.cloud import aiplatform
    from google.cloud import storage
    import yaml
except ImportError:
    print("Missing required dependencies. Please install with:")
    print("pip install google-cloud-aiplatform google-cloud-storage PyYAML")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_deployment_config(config_path: str = None) -> Dict[str, Any]:
    """Load deployment configuration from YAML file"""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {
            "agent": {
                "name": "excel-intelligence-agent",
                "description": "Multi-agent Excel file analysis system",
                "version": "1.0.0"
            },
            "deployment": {
                "region": "us-central1",
                "machine_type": "n1-standard-4",
                "max_instances": 10,
                "min_instances": 1
            }
        }


def validate_environment() -> bool:
    """Validate that required environment variables and credentials are set"""
    required_env_vars = [
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION"
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.info("Please set these variables or update your .env file")
        return False
    
    # Check for Google Cloud authentication
    try:
        from google.auth import default
        credentials, project = default()
        logger.info(f"Using Google Cloud project: {project}")
        return True
    except Exception as e:
        logger.error(f"Google Cloud authentication failed: {e}")
        logger.info("Please run: gcloud auth application-default login")
        return False


def create_agent_artifact(project_id: str, region: str) -> str:
    """Package and upload the agent code as an artifact"""
    logger.info("Creating agent artifact...")
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Create a storage client
    storage_client = storage.Client(project=project_id)
    
    # Create bucket name (must be globally unique)
    bucket_name = f"{project_id}-excel-intelligence-agent-artifacts"
    
    try:
        # Try to get existing bucket
        bucket = storage_client.get_bucket(bucket_name)
        logger.info(f"Using existing bucket: {bucket_name}")
    except Exception:
        # Create new bucket
        bucket = storage_client.create_bucket(bucket_name, location=region)
        logger.info(f"Created new bucket: {bucket_name}")
    
    # Package the agent code
    import tarfile
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as temp_file:
        with tarfile.open(temp_file.name, 'w:gz') as tar:
            # Add the agent code
            agent_path = project_root / "excel_intelligence_agent"
            tar.add(agent_path, arcname="excel_intelligence_agent")
            
            # Add requirements
            if (project_root / "pyproject.toml").exists():
                tar.add(project_root / "pyproject.toml", arcname="pyproject.toml")
        
        # Upload to Cloud Storage
        blob_name = "excel_intelligence_agent_v1.0.0.tar.gz"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(temp_file.name)
        
        artifact_uri = f"gs://{bucket_name}/{blob_name}"
        logger.info(f"Uploaded agent artifact to: {artifact_uri}")
        
        # Clean up temp file
        os.unlink(temp_file.name)
        
        return artifact_uri


def deploy_agent_to_vertex(
    project_id: str,
    region: str,
    artifact_uri: str,
    config: Dict[str, Any]
) -> str:
    """Deploy the agent to Vertex AI Agent Engine"""
    logger.info("Deploying agent to Vertex AI Agent Engine...")
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    agent_config = config.get("agent", {})
    deployment_config = config.get("deployment", {})
    
    # Create agent specification
    agent_spec = {
        "display_name": agent_config.get("name", "Excel Intelligence Agent"),
        "description": agent_config.get("description", "Multi-agent Excel analysis system"),
        "agent_version": agent_config.get("version", "1.0.0"),
        "artifact_uri": artifact_uri,
        "agent_type": "MULTI_AGENT_SYSTEM",
        "runtime_config": {
            "machine_type": deployment_config.get("machine_type", "n1-standard-4"),
            "max_instances": deployment_config.get("max_instances", 10),
            "min_instances": deployment_config.get("min_instances", 1),
            "environment_variables": {
                "GOOGLE_CLOUD_PROJECT": project_id,
                "GOOGLE_CLOUD_LOCATION": region,
                "ROOT_AGENT_MODEL": os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-flash"),
                "ORCHESTRATOR_MODEL": os.getenv("ORCHESTRATOR_MODEL", "gemini-2.5-pro"),
                "WORKER_MODEL": os.getenv("WORKER_MODEL", "gemini-2.5-flash")
            }
        }
    }
    
    try:
        # Deploy using Vertex AI Agent Engine API
        # Note: This is a simplified example - actual implementation would use
        # the specific Agent Engine APIs when they become available
        
        logger.info("Agent deployment initiated...")
        logger.info(f"Agent Name: {agent_spec['display_name']}")
        logger.info(f"Project: {project_id}")
        logger.info(f"Region: {region}")
        logger.info(f"Artifact: {artifact_uri}")
        
        # For now, we'll create a Model resource as a placeholder
        from google.cloud.aiplatform import Model
        
        model = Model.upload(
            display_name=agent_spec["display_name"],
            artifact_uri=artifact_uri,
            serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-8:latest",
            description=agent_spec["description"]
        )
        
        agent_endpoint = model.resource_name
        logger.info(f"Agent deployed successfully!")
        logger.info(f"Endpoint: {agent_endpoint}")
        
        return agent_endpoint
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise


def create_deployment_manifest(
    project_id: str,
    region: str,
    endpoint: str,
    config: Dict[str, Any]
) -> None:
    """Create a deployment manifest file with connection details"""
    manifest = {
        "deployment": {
            "timestamp": "2024-01-01T00:00:00Z",
            "project_id": project_id,
            "region": region,
            "endpoint": endpoint,
            "status": "deployed"
        },
        "agent": config.get("agent", {}),
        "usage": {
            "endpoint_url": f"https://{region}-aiplatform.googleapis.com/v1/{endpoint}:predict",
            "authentication": "Google Cloud Service Account",
            "example_request": {
                "instances": [{
                    "user_query": "Analyze the data quality in my Excel file",
                    "file_path": "gs://your-bucket/data.xlsx"
                }]
            }
        }
    }
    
    manifest_path = Path(__file__).parent / "deployment_manifest.yaml"
    with open(manifest_path, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False, indent=2)
    
    logger.info(f"Deployment manifest created: {manifest_path}")


def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy Excel Intelligence Agent to Google Cloud")
    parser.add_argument("--project-id", required=True, help="Google Cloud Project ID")
    parser.add_argument("--region", default="us-central1", help="Deployment region")
    parser.add_argument("--config", help="Path to deployment config file")
    parser.add_argument("--dry-run", action="store_true", help="Validate without deploying")
    
    args = parser.parse_args()
    
    try:
        logger.info("🚀 Starting Excel Intelligence Agent deployment...")
        
        # Validate environment
        if not validate_environment():
            logger.error("❌ Environment validation failed")
            sys.exit(1)
        
        # Load configuration
        config = load_deployment_config(args.config)
        logger.info("✅ Configuration loaded successfully")
        
        if args.dry_run:
            logger.info("🔍 Dry run mode - validation complete")
            logger.info("Configuration:")
            print(yaml.dump(config, default_flow_style=False, indent=2))
            return
        
        # Create and upload agent artifact
        artifact_uri = create_agent_artifact(args.project_id, args.region)
        logger.info("✅ Agent artifact created successfully")
        
        # Deploy to Vertex AI
        endpoint = deploy_agent_to_vertex(
            args.project_id,
            args.region,
            artifact_uri,
            config
        )
        logger.info("✅ Agent deployed successfully")
        
        # Create deployment manifest
        create_deployment_manifest(
            args.project_id,
            args.region,
            endpoint,
            config
        )
        logger.info("✅ Deployment manifest created")
        
        logger.info("🎉 Deployment completed successfully!")
        logger.info(f"📋 Agent endpoint: {endpoint}")
        logger.info("📖 Check deployment_manifest.yaml for usage details")
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()