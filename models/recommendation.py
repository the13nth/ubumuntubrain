from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Metadata:
    """Metadata for a recommendation"""
    source: str
    type: str
    url: Optional[str] = None
    timestamp: Optional[str] = None
    additional: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Metadata':
        """Create Metadata object from dictionary"""
        additional = {k: v for k, v in data.items() 
                      if k not in ['source', 'type', 'url', 'timestamp']}
        
        return cls(
            source=data.get('source', 'unknown'),
            type=data.get('type', 'general'),
            url=data.get('url'),
            timestamp=data.get('timestamp'),
            additional=additional
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'source': self.source,
            'type': self.type
        }
        
        if self.url:
            result['url'] = self.url
        if self.timestamp:
            result['timestamp'] = self.timestamp
            
        # Add additional fields
        result.update(self.additional)
        
        return result

@dataclass
class Recommendation:
    """Recommendation model with text, metadata and score"""
    text: str
    metadata: Metadata
    score: float
    id: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Recommendation':
        """Create Recommendation object from dictionary"""
        metadata = Metadata.from_dict(data.get('metadata', {}))
        
        return cls(
            text=data.get('text', ''),
            metadata=metadata,
            score=data.get('score', 0.0),
            id=data.get('id')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'text': self.text,
            'metadata': self.metadata.to_dict(),
            'score': self.score
        }
        
        if self.id:
            result['id'] = self.id
            
        return result

@dataclass
class RecommendationSet:
    """Collection of recommendations with metadata"""
    recommendations: List[Recommendation]
    timestamp: datetime = field(default_factory=datetime.now)
    query: Optional[str] = None
    
    @property
    def count(self) -> int:
        """Get number of recommendations"""
        return len(self.recommendations)
    
    def add(self, recommendation: Recommendation) -> None:
        """Add a recommendation to the set"""
        self.recommendations.append(recommendation)
        
    def sort(self) -> None:
        """Sort recommendations by score (highest first)"""
        self.recommendations.sort(key=lambda x: x.score, reverse=True)
        
    def filter(self, min_score: float = 0.0, 
               source: Optional[str] = None, 
               type_filter: Optional[str] = None) -> 'RecommendationSet':
        """Filter recommendations by criteria"""
        filtered = []
        
        for rec in self.recommendations:
            # Apply score filter
            if rec.score < min_score:
                continue
                
            # Apply source filter if specified
            if source and rec.metadata.source != source:
                continue
                
            # Apply type filter if specified
            if type_filter and rec.metadata.type != type_filter:
                continue
                
            filtered.append(rec)
            
        return RecommendationSet(
            recommendations=filtered,
            timestamp=self.timestamp,
            query=self.query
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecommendationSet':
        """Create RecommendationSet from dictionary"""
        recommendations = []
        
        for rec_data in data.get('recommendations', []):
            recommendations.append(Recommendation.from_dict(rec_data))
            
        timestamp = datetime.fromisoformat(data.get('timestamp')) if 'timestamp' in data else datetime.now()
        
        return cls(
            recommendations=recommendations,
            timestamp=timestamp,
            query=data.get('query')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'recommendations': [rec.to_dict() for rec in self.recommendations],
            'count': self.count,
            'timestamp': self.timestamp.isoformat()
        }
        
        if self.query:
            result['query'] = self.query
            
        return result 