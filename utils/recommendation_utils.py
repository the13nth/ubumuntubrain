from typing import Dict, List, Any, Optional
import hashlib
import json
import os
from datetime import datetime

def generate_document_id(metadata: Dict[str, Any]) -> str:
    """
    Generate a unique document ID based on metadata.
    
    Args:
        metadata: Document metadata
        
    Returns:
        Unique document ID as string
    """
    # Create a string from metadata
    metadata_str = json.dumps(metadata, sort_keys=True)
    # Generate MD5 hash
    return hashlib.md5(metadata_str.encode()).hexdigest()

def format_recommendation(result: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Format a single recommendation result from ChromaDB.
    
    Args:
        result: Result from ChromaDB query
        index: Index of result in query results
        
    Returns:
        Formatted recommendation
    """
    try:
        # Extract data from result
        metadata = result['metadatas'][0][index]
        document = result['documents'][0][index]
        distance = result['distances'][0][index]
        
        # Create recommendation object
        recommendation = {
            "text": document,
            "metadata": metadata,
            "score": 1.0 - distance  # Convert distance to similarity score
        }
        
        # Add additional fields if available in metadata
        if "source" in metadata:
            recommendation["source"] = metadata["source"]
        if "type" in metadata:
            recommendation["type"] = metadata["type"]
            
        return recommendation
    except (IndexError, KeyError) as e:
        # Handle missing data
        print(f"Error formatting recommendation: {str(e)}")
        return {
            "text": "Error retrieving recommendation",
            "score": 0.0,
            "metadata": {}
        }

def filter_recommendations(results: Dict[str, Any], threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Filter recommendations by similarity threshold and format them.
    
    Args:
        results: Results from ChromaDB query
        threshold: Minimum similarity score (1 - distance)
        
    Returns:
        List of filtered and formatted recommendations
    """
    recommendations = []
    
    if not results or 'distances' not in results or not results['distances']:
        return recommendations
    
    # Get the number of results
    num_results = len(results['distances'][0])
    
    # Process each result
    for i in range(num_results):
        # Calculate similarity score
        distance = results['distances'][0][i]
        similarity = 1.0 - distance
        
        # Apply threshold filter
        if similarity >= threshold:
            recommendation = format_recommendation(results, i)
            recommendations.append(recommendation)
    
    # Sort by score (highest first)
    return sorted(recommendations, key=lambda x: x['score'], reverse=True)

def save_recommendations_to_file(recommendations: List[Dict[str, Any]], 
                                file_path: str,
                                query: Optional[str] = None) -> bool:
    """
    Save recommendations to a JSON file with timestamp and query info.
    
    Args:
        recommendations: List of recommendation objects
        file_path: Path to save the file
        query: Optional query text that generated these recommendations
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Create output object
        output = {
            "timestamp": datetime.now().isoformat(),
            "count": len(recommendations),
            "recommendations": recommendations
        }
        
        # Add query if provided
        if query:
            output["query"] = query
            
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
        return True
    except Exception as e:
        print(f"Error saving recommendations to file: {str(e)}")
        return False 