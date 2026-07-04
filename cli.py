import click
import logging
from app.core.container import Container
from app.api import create_app
from app.services.system_service import SystemService

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("CLI")

@click.group()
def cli():
    """Pi Vision AI - Computer Vision Framework CLI"""
    # Initialize the container for all CLI commands
    Container.get_instance()

@cli.command()
@click.option('--host', default='127.0.0.1', help='Host to bind to')
@click.option('--port', default=5000, help='Port to bind to')
def run(host, port):
    """Run the REST API server."""
    container = Container.get_instance()
    
    # Start background threads
    container.camera_manager.start()
        
    app = create_app()
    logger.info(f"Starting Pi Vision AI server on {host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)

@cli.command()
def doctor():
    """Run system diagnostics."""
    service = SystemService()
    status = service.get_status()
    click.echo("\n--- System Diagnostics ---")
    click.echo(f"API Mode: {status['session'].get('api_key_mode', 'Local')}")
    click.echo(f"Active Backend: {status['backend']['name'] if status.get('backend') else 'None'}")
    click.echo(f"Camera Connected: {status['camera']['is_connected']}")
    click.echo(f"RAM Usage: {status['performance']['ram_usage_gb']} GB")
    click.echo(f"CPU Usage: {status['performance']['cpu_usage_percent']} %")
    click.echo("--------------------------\n")

@cli.command()
def models():
    """List available AI models."""
    container = Container.get_instance()
    models = container.model_registry.list_models()
    click.echo(f"\nDiscovered {len(models)} models:")
    for m in models:
        click.echo(f"- {m.name} ({m.version}) [{m.framework}]")
    click.echo()

if __name__ == '__main__':
    cli()
