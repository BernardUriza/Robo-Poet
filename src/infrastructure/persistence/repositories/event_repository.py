"""
SQLAlchemy implementation of EventRepository.

Concrete implementation using SQLAlchemy for domain event storage.
"""

from typing import List
from sqlalchemy.orm import Session

from src.domain.events.training_events import DomainEvent
from src.infrastructure.persistence.sqlalchemy_models import EventORM


class SQLAlchemyEventRepository:
    """SQLAlchemy implementation of the event repository."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def add(self, event: DomainEvent) -> None:
        """Store a domain event."""
        orm_event = EventORM.from_domain(event)
        self.session.add(orm_event)
        # Note: commit is handled by UnitOfWork
    
    def get_by_aggregate_id(self, aggregate_id: str) -> List[dict]:
        """Get all events for an aggregate."""
        orm_events = self.session.query(EventORM).filter(
            EventORM.aggregate_id == aggregate_id
        ).order_by(EventORM.timestamp.asc()).all()
        
        return [event.to_domain() for event in orm_events]
    
    def get_by_type(self, event_type: str) -> List[dict]:
        """Get all events of a specific type."""
        orm_events = self.session.query(EventORM).filter(
            EventORM.event_type == event_type
        ).order_by(EventORM.timestamp.desc()).all()
        
        return [event.to_domain() for event in orm_events]
    
    def get_recent(self, limit: int = 100) -> List[dict]:
        """Get most recent events."""
        orm_events = self.session.query(EventORM).order_by(
            EventORM.timestamp.desc()
        ).limit(limit).all()
        
        return [event.to_domain() for event in orm_events]
    
    def count(self) -> int:
        """Get total count of events."""
        return self.session.query(EventORM).count()
    
    def delete_old_events(self, days: int = 30) -> int:
        """Delete events older than specified days."""
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = self.session.query(EventORM).filter(
            EventORM.timestamp < cutoff_date
        ).count()
        
        self.session.query(EventORM).filter(
            EventORM.timestamp < cutoff_date
        ).delete()
        
        return deleted_count