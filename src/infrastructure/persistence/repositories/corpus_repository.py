"""
SQLAlchemy implementation of CorpusRepository.

Concrete implementation using SQLAlchemy for corpus data access.
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.domain.entities.text_corpus import TextCorpus
from src.infrastructure.persistence.sqlalchemy_models import CorpusORM


class SQLAlchemyCorpusRepository:
    """SQLAlchemy implementation of the corpus repository."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def add(self, corpus: TextCorpus) -> None:
        """Add a new corpus to the repository."""
        orm_corpus = CorpusORM.from_domain(corpus)
        self.session.add(orm_corpus)
        # Note: commit is handled by UnitOfWork
    
    def get(self, corpus_id: str) -> Optional[TextCorpus]:
        """Get corpus by ID."""
        orm_corpus = self.session.query(CorpusORM).filter(
            CorpusORM.id == corpus_id
        ).first()
        
        return orm_corpus.to_domain() if orm_corpus else None
    
    def list(self, tags: Optional[List[str]] = None, limit: Optional[int] = None) -> List[TextCorpus]:
        """List corpora with optional filtering."""
        query = self.session.query(CorpusORM)
        
        # Filter by tags if provided
        if tags:
            # JSON contains any of the specified tags
            for tag in tags:
                query = query.filter(
                    func.json_contains(CorpusORM.tags, f'"{tag}"')
                )
        
        # Apply limit
        if limit:
            query = query.limit(limit)
        
        # Order by creation date (newest first)
        query = query.order_by(CorpusORM.created_at.desc())
        
        orm_corpora = query.all()
        return [corpus.to_domain() for corpus in orm_corpora]
    
    def update(self, corpus: TextCorpus) -> None:
        """Update existing corpus."""
        orm_corpus = self.session.query(CorpusORM).filter(
            CorpusORM.id == corpus.id
        ).first()
        
        if not orm_corpus:
            raise ValueError(f"Corpus with ID {corpus.id} not found")
        
        # Update fields
        orm_corpus.name = corpus.name
        orm_corpus.content = corpus.content
        orm_corpus.source_path = corpus.source_path
        orm_corpus.updated_at = corpus.updated_at
        orm_corpus.preprocessed = corpus.preprocessed
        orm_corpus.vocabulary_size = corpus.vocabulary_size
        orm_corpus.token_count = corpus.token_count
        orm_corpus.sequence_count = corpus.sequence_count
        orm_corpus.language = corpus.language
        orm_corpus.encoding = corpus.encoding
        orm_corpus.tags = corpus.tags
    
    def delete(self, corpus_id: str) -> None:
        """Delete corpus by ID."""
        orm_corpus = self.session.query(CorpusORM).filter(
            CorpusORM.id == corpus_id
        ).first()
        
        if orm_corpus:
            self.session.delete(orm_corpus)
    
    def find_by_name(self, name: str) -> Optional[TextCorpus]:
        """Find corpus by name."""
        orm_corpus = self.session.query(CorpusORM).filter(
            CorpusORM.name == name
        ).first()
        
        return orm_corpus.to_domain() if orm_corpus else None
    
    def find_by_content_hash(self, content_hash: str) -> Optional[TextCorpus]:
        """Find corpus by content hash to detect duplicates."""
        # Note: We'd need to add content_hash column to database
        # For now, we'll check in-memory (inefficient but functional)
        orm_corpora = self.session.query(CorpusORM).all()
        
        for orm_corpus in orm_corpora:
            corpus = orm_corpus.to_domain()
            if corpus.content_hash() == content_hash:
                return corpus
        
        return None
    
    def count(self) -> int:
        """Get total count of corpora."""
        return self.session.query(CorpusORM).count()
    
    def find_preprocessed(self) -> List[TextCorpus]:
        """Find all preprocessed corpora."""
        orm_corpora = self.session.query(CorpusORM).filter(
            CorpusORM.preprocessed == True
        ).all()
        
        return [corpus.to_domain() for corpus in orm_corpora]
    
    def find_by_language(self, language: str) -> List[TextCorpus]:
        """Find corpora by language."""
        orm_corpora = self.session.query(CorpusORM).filter(
            CorpusORM.language == language
        ).all()
        
        return [corpus.to_domain() for corpus in orm_corpora]