"""
CLI Controller - Clean Architecture Entry Point

Replaces the monolithic orchestrator with proper separation of concerns.
Handles only CLI-specific concerns and delegates business logic to application services.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from src.application.services.cli_service import CLIApplicationService, CLIConfig
from src.application.services.training_service import TrainingService
from src.application.services.generation_service import GenerationService
from src.infrastructure.container import Container
from src.config.settings import get_cached_settings
from src.core.exceptions import RoboPoetError


class CLIController:
    """
    CLI Controller following Clean Architecture principles.
    
    Responsibilities:
    - Parse command line arguments
    - Initialize dependency injection container
    - Delegate to application services
    - Handle CLI-specific error formatting
    - Manage console output
    """
    
    def __init__(self):
        self.settings = get_cached_settings()
        self.container = Container()
        self.logger = logging.getLogger(__name__)
        
        # Initialize application service
        self.cli_service = CLIApplicationService(
            training_service=self.container.training_service(),
            generation_service=self.container.generation_service(),
            settings=self.settings
        )
    
    def run(self, args: Optional[list] = None) -> int:
        """
        Main entry point for CLI execution.
        
        Args:
            args: Command line arguments (None uses sys.argv)
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            # Parse arguments
            parsed_args = self._parse_arguments(args)
            config = self._create_cli_config(parsed_args)
            
            # Route to appropriate handler
            if parsed_args.text and not parsed_args.generate:
                return self._handle_training(config)
            elif parsed_args.generate:
                return self._handle_generation(config)
            else:
                return self._handle_interactive_mode()
                
        except KeyboardInterrupt:
            self._print_info("\\n🛑 Operation cancelled by user")
            return 130  # Standard exit code for Ctrl+C
        except RoboPoetError as e:
            self._print_error(f"Robo-Poet Error: {e.message}")
            if e.recovery_suggestions:
                for suggestion in e.recovery_suggestions:
                    self._print_info(f"💡 Suggestion: {suggestion}")
            return 1
        except Exception as e:
            self._print_error(f"Unexpected error: {e}")
            self.logger.exception("Unexpected error in CLI")
            return 1
    
    def _handle_training(self, config: CLIConfig) -> int:
        """Handle training workflow."""
        try:
            self._print_header("🎓 ROBO-POET TRAINING MODE")
            self._print_info(f"📁 Training on: {config.text_file}")
            self._print_info(f"🔄 Epochs: {config.epochs}")
            
            # Show system status
            status = self.cli_service.get_system_status()
            gpu_status = "✅ Available" if status.get("gpu_available") else "❌ Not Available" 
            self._print_info(f"🎯 GPU: {gpu_status}")
            
            # Train model
            model_id = self.cli_service.train_model_from_file(config)
            
            self._print_success(f"🎉 Training completed! Model ID: {model_id}")
            
            # Show model location
            models_dir = status.get("storage_paths", {}).get("models", "./models")
            self._print_info(f"📍 Model saved to: {models_dir}")
            
            return 0
            
        except Exception as e:
            self._print_error(f"Training failed: {e}")
            return 1
    
    def _handle_generation(self, config: CLIConfig) -> int:
        """Handle text generation workflow."""
        try:
            self._print_header("🎨 ROBO-POET GENERATION MODE")
            self._print_info(f"🤖 Model: {config.model_file}")
            self._print_info(f"🌱 Seed: '{config.seed_text}'")
            self._print_info(f"🌡️ Temperature: {config.temperature}")
            self._print_info(f"📏 Length: {config.length}")
            
            # Generate text
            result = self.cli_service.generate_text(config)
            
            # Display result
            self._print_header("📝 GENERATED TEXT")
            print(result)
            self._print_success("\\n✅ Generation completed!")
            
            return 0
            
        except Exception as e:
            self._print_error(f"Generation failed: {e}")
            return 1
    
    def _handle_interactive_mode(self) -> int:
        """Handle interactive academic menu mode."""
        try:
            # Import menu system dynamically to avoid circular imports
            from src.interface.menu_system import AcademicMenuSystem
            
            self._print_header("🎓 ROBO-POET ACADEMIC INTERFACE")
            
            # Show system status
            status = self.cli_service.get_system_status()
            self._print_system_status(status)
            
            # Initialize and run menu system
            menu_system = AcademicMenuSystem()
            menu_system.run_main_loop()
            
            return 0
            
        except ImportError:
            self._print_error("Academic menu system not available")
            self._print_info("💡 Try using direct mode: --text <file> or --generate <model>")
            return 1
        except Exception as e:
            self._print_error(f"Interactive mode failed: {e}")
            return 1
    
    def _parse_arguments(self, args: Optional[list] = None) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            prog="robo-poet",
            description="🎓 Robo-Poet Academic Neural Text Generation Framework",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  robo_poet.py                                    # Interactive academic interface
  robo_poet.py --text archivo.txt --epochs 20    # Direct training
  robo_poet.py --generate modelo.keras           # Direct generation
  robo_poet.py --generate modelo.keras --seed "The power" --temp 0.8
            """
        )
        
        parser.add_argument(
            "--text",
            help="Text file for training",
            type=str
        )
        
        parser.add_argument(
            "--epochs", 
            help="Number of training epochs (default: from config)",
            type=int
        )
        
        parser.add_argument(
            "--generate",
            help="Model file for text generation", 
            type=str
        )
        
        parser.add_argument(
            "--seed",
            help="Seed text for generation (default: 'The power of')",
            type=str,
            default="The power of"
        )
        
        parser.add_argument(
            "--temp", "--temperature",
            help="Temperature for generation (default: 0.8)",
            type=float,
            default=0.8,
            dest="temperature"
        )
        
        parser.add_argument(
            "--length",
            help="Length of generated text (default: 200)",
            type=int,
            default=200
        )
        
        return parser.parse_args(args)
    
    def _create_cli_config(self, args: argparse.Namespace) -> CLIConfig:
        """Create CLI configuration from parsed arguments."""
        return CLIConfig(
            text_file=args.text,
            epochs=args.epochs,
            model_file=args.generate,
            seed_text=args.seed,
            temperature=args.temperature,
            length=args.length
        )
    
    def _print_header(self, title: str) -> None:
        """Print formatted header."""
        print("=" * 70)
        print(title.center(70))
        print("=" * 70)
    
    def _print_success(self, message: str) -> None:
        """Print success message."""
        print(f"✅ {message}")
    
    def _print_error(self, message: str) -> None:
        """Print error message."""
        print(f"❌ {message}", file=sys.stderr)
    
    def _print_warning(self, message: str) -> None:
        """Print warning message."""
        print(f"⚠️ {message}")
    
    def _print_info(self, message: str) -> None:
        """Print info message."""
        print(f"ℹ️ {message}")
    
    def _print_system_status(self, status: dict) -> None:
        """Print formatted system status."""
        print("🔍 SYSTEM STATUS:")
        print("-" * 50)
        
        gpu_status = "✅ Available" if status.get("gpu_available") else "❌ Not Available"
        print(f"🎯 GPU: {gpu_status}")
        
        if "model_config" in status:
            model_cfg = status["model_config"]
            print(f"🧠 Model Config: {model_cfg.get('vocab_size', 'N/A')} vocab, {model_cfg.get('lstm_units', 'N/A')} LSTM units")
        
        if "training_config" in status:
            train_cfg = status["training_config"]
            print(f"🎓 Training: {train_cfg.get('default_epochs', 'N/A')} epochs, batch {train_cfg.get('default_batch_size', 'N/A')}")
        
        print("-" * 50)


def main() -> int:
    """Main entry point for CLI."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run CLI controller
    controller = CLIController()
    return controller.run()


if __name__ == "__main__":
    sys.exit(main())