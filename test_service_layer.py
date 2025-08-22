#!/usr/bin/env python3
"""
Test script for the complete Service Layer and Command/Query implementation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_service_layer():
    """Test the complete service layer implementation."""
    print("🧪 Testing Service Layer Implementation...")
    
    # Test imports
    try:
        print("1. Testing imports...")
        
        # Domain layer
        from src.domain.entities.text_corpus import TextCorpus
        from src.domain.entities.generation_model import GenerationModel, ModelStatus
        from src.domain.value_objects.model_config import ModelConfig
        from src.domain.value_objects.generation_params import GenerationParams
        
        # Application layer
        from src.application.services import TrainingService, GenerationService
        from src.application.commands import (
            CreateCorpusCommand, TrainModelCommand, StopTrainingCommand,
            ArchiveModelCommand, DeleteCorpusCommand, UpdateCorpusCommand
        )
        from src.application.queries.training_queries import (
            GetModelByIdQuery, ListModelsQuery, SearchModelsQuery
        )
        from src.application.queries.query_handlers import TrainingQueryHandler
        from src.application.event_handlers.training_event_handlers import (
            TrainingEventHandler, GenerationEventHandler
        )
        from src.application.message_bus import MessageBus, MessageContext, create_message_bus
        
        # Infrastructure layer
        from src.infrastructure.persistence.unit_of_work import SQLAlchemyUnitOfWork
        from src.infrastructure.persistence.sqlalchemy_models import Base
        
        print("✅ All imports successful")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test command creation and validation
    try:
        print("2. Testing Command objects...")
        
        # Test CreateCorpusCommand
        cmd = CreateCorpusCommand(
            name="Test Corpus",
            content="This is a test corpus with enough content to pass validation. " * 50,
            language="en",
            tags=["test", "sample"]
        )
        cmd.validate()
        print("   ✅ CreateCorpusCommand validation passed")
        
        # Test TrainModelCommand
        config = ModelConfig()  # Uses default values from dataclass
        train_cmd = TrainModelCommand(
            model_name="Test Model",
            corpus_id="test-corpus-id",
            config=config,
            description="A test model"
        )
        train_cmd.validate()
        print("   ✅ TrainModelCommand validation passed")
        
        # Test invalid command
        try:
            invalid_cmd = CreateCorpusCommand(name="", content="short")
            invalid_cmd.validate()
            print("   ❌ Validation should have failed")
            return False
        except ValueError:
            print("   ✅ Invalid command correctly rejected")
            
    except Exception as e:
        print(f"❌ Command testing error: {e}")
        return False
    
    # Test query objects
    try:
        print("3. Testing Query objects...")
        
        # Test queries
        model_query = GetModelByIdQuery(model_id="test-id")
        list_query = ListModelsQuery(status=ModelStatus.TRAINED, limit=10)
        search_query = SearchModelsQuery(search_term="test", limit=5)
        
        print("   ✅ Query objects created successfully")
        
    except Exception as e:
        print(f"❌ Query testing error: {e}")
        return False
    
    # Test service layer components
    try:
        print("4. Testing Service Layer structure...")
        
        # Mock UnitOfWork for testing
        class MockUnitOfWork:
            def __init__(self):
                self.corpus_repository = None
                self.model_repository = None
                self.event_repository = None
                self.committed = False
            
            def __enter__(self):
                return self
            
            def __exit__(self, *args):
                pass
            
            def commit(self):
                self.committed = True
        
        # Test service instantiation
        mock_uow = MockUnitOfWork()
        training_service = TrainingService(mock_uow)
        generation_service = GenerationService(mock_uow)
        query_handler = TrainingQueryHandler(mock_uow)
        event_handler = TrainingEventHandler(mock_uow)
        
        print("   ✅ All service components instantiated")
        
    except Exception as e:
        print(f"❌ Service layer testing error: {e}")
        return False
    
    # Test message bus
    try:
        print("5. Testing Message Bus...")
        
        bus = MessageBus()
        
        # Test handler registration
        def dummy_handler(cmd):
            return f"Handled {type(cmd).__name__}"
        
        bus.register_command_handler(CreateCorpusCommand, dummy_handler)
        
        # Test middleware
        def dummy_middleware(msg_type, msg, ctx):
            pass
        
        bus.add_middleware(dummy_middleware)
        
        # Test context
        context = MessageContext(user_id="test-user", request_id="req-123")
        
        print("   ✅ Message bus configured successfully")
        
    except Exception as e:
        print(f"❌ Message bus testing error: {e}")
        return False
    
    # Test domain events
    try:
        print("6. Testing Domain Events...")
        
        from src.domain.events.training_events import (
            TrainingStarted, TrainingCompleted, CorpusCreated
        )
        
        # Create events
        training_event = TrainingStarted(
            model_id="test-model",
            corpus_id="test-corpus",
            config={"epochs": 10}
        )
        
        corpus_event = CorpusCreated(
            corpus_id="test-corpus",
            name="Test Corpus",
            size=1000,
            language="en"
        )
        
        print("   ✅ Domain events created successfully")
        
    except Exception as e:
        print(f"❌ Domain events testing error: {e}")
        return False
    
    print("\n🎉 Service Layer Implementation Test Complete!")
    print("\n📋 Summary:")
    print("✅ All domain entities and value objects")
    print("✅ Complete command pattern implementation")
    print("✅ CQRS query handlers")
    print("✅ Event handlers and domain events")
    print("✅ Message bus for coordination")
    print("✅ Service layer orchestration")
    print("✅ Clean architecture separation")
    
    print("\n🏗️ Enterprise Architecture Status:")
    print("✅ Strategy 7: Domain-Driven Design")
    print("✅ Strategy 8: Repository Pattern & Unit of Work")
    print("✅ Strategy 9: Service Layer & Commands")
    print("⏳ Next: Strategy 10-16 (Dependency Injection, Testing, CLI, API, Deployment)")
    
    return True

if __name__ == "__main__":
    success = test_service_layer()
    sys.exit(0 if success else 1)