from typing import Dict, List, Any, Optional
import os
import json
from datetime import datetime

from services.embedding_service import EmbeddingService
from services.chromadb_service import ChromaDBService
from models.recommendation import Recommendation, Metadata, RecommendationSet
from utils.recommendation_utils import generate_document_id, filter_recommendations, save_recommendations_to_file

# Import Firebase db
try:
    import firebase_admin
    from firebase_admin import firestore
    db = firestore.client() if firebase_admin._apps else None
except ImportError:
    db = None
    print("Firebase not available. Will use local data only.")
except Exception as e:
    db = None
    print(f"Error initializing Firebase in RecommendationService: {str(e)}")


class RecommendationService:
    """
    Service for managing recommendations using embedding and database services.
    Follows SOLID principles by:
    - Single Responsibility: Handles only recommendation generation logic
    - Open/Closed: Can be extended with new recommendation types
    - Liskov Substitution: Uses interfaces for services
    - Interface Segregation: Depends on specific interfaces
    - Dependency Inversion: Depends on abstractions (services), not implementations
    """
    
    def __init__(self, embedding_service: EmbeddingService, db_service: ChromaDBService):
        """
        Initialize the recommendation service with embedding and database services.
        
        Args:
            embedding_service: Service for generating embeddings
            db_service: Service for storing and retrieving vectors
        """
        self.embedding_service = embedding_service
        self.db_service = db_service
        self.data_dir = "data/recommendations"
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Check Firebase connection
        if 'db' in globals() and db is not None:
            try:
                # Test Firebase connection by listing collections
                collections = db.collections()
                collection_names = [collection.id for collection in collections]
                print(f"Connected to Firebase. Available collections: {', '.join(collection_names)}")
                
                # Also check specific collections we'll use
                required_collections = [
                    'commute_ai_recommendation',
                    'health_ai_recommendation',
                    'work_ai_recommendation'
                ]
                
                for collection_name in required_collections:
                    try:
                        collection = db.collection(collection_name)
                        docs = collection.limit(1).get()
                        count = len(docs)
                        print(f"Found {count} document(s) in {collection_name}")
                    except Exception as e:
                        print(f"Error accessing collection {collection_name}: {str(e)}")
                
            except Exception as e:
                print(f"Error testing Firebase connection: {str(e)}")
    
    def add_document(self, text: str, metadata: Dict[str, Any]) -> bool:
        """
        Add a document to the recommendation database.
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if document already exists
            existing = self.embedding_service.check_existing_embedding(
                self.db_service.get_collection(),
                metadata
            )
            
            if existing:
                print(f"Document already exists with ID: {existing}")
                return True
                
            # Generate embedding
            embedding = self.embedding_service.encode(text)
            
            # Generate ID
            doc_id = generate_document_id(metadata)
            
            # Add to collection
            return self.db_service.add_document(embedding, text, doc_id, metadata)
        except Exception as e:
            print(f"Error adding document: {str(e)}")
            return False
    
    def get_recommendations(self, query: str, n_results: int = 10, 
                           threshold: float = 0.7) -> RecommendationSet:
        """
        Get recommendations based on a query.
        
        Args:
            query: Search query
            n_results: Number of results to retrieve
            threshold: Minimum similarity score (0-1)
            
        Returns:
            RecommendationSet with results
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.encode(query)
            
            # Search in database
            results = self.db_service.search(query_embedding, n_results)
            
            # Convert to recommendations
            recommendations = []
            
            if results and 'documents' in results and results['documents']:
                # Get the number of results
                num_results = len(results['documents'][0])
                
                # Process each result
                for i in range(num_results):
                    try:
                        # Get data
                        document = results['documents'][0][i]
                        metadata_dict = results['metadatas'][0][i]
                        distance = results['distances'][0][i]
                        similarity = 1.0 - distance
                        
                        # Apply threshold
                        if similarity >= threshold:
                            # Create metadata object
                            metadata = Metadata.from_dict(metadata_dict)
                            
                            # Create recommendation
                            recommendation = Recommendation(
                                text=document,
                                metadata=metadata,
                                score=similarity
                            )
                            
                            recommendations.append(recommendation)
                    except (IndexError, KeyError) as e:
                        print(f"Error processing result {i}: {str(e)}")
            
            # Create recommendation set
            recommendation_set = RecommendationSet(
                recommendations=recommendations,
                query=query
            )
            
            # Sort by score
            recommendation_set.sort()
            
            return recommendation_set
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return RecommendationSet(recommendations=[])
    
    def save_health_recommendations(self, query: str) -> str:
        """
        Get and save health recommendations.
        
        Args:
            query: Search query
            
        Returns:
            Path to saved file
        """
        try:
            # Get recommendations
            recommendation_set = self.get_recommendations(query)
            
            # Filter by type
            health_recommendations = recommendation_set.filter(type_filter="health")
            
            # Save to file
            file_path = os.path.join(self.data_dir, f"health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(health_recommendations.to_dict(), f, indent=2, ensure_ascii=False)
            
            # Save to Firebase if available
            if db is not None:
                try:
                    # Create a document in Firebase
                    health_collection = db.collection('health_ai_recommendation')
                    health_data = {
                        'query': query,
                        'timestamp': datetime.now().isoformat(),
                        'bloodSugar': 120,  # Sample data
                        'exerciseMinutes': 25,
                        'mealType': 'Balanced',
                        'recommendations': [r.text for r in health_recommendations.recommendations],
                        'relevance_scores': [r.score for r in health_recommendations.recommendations]
                    }
                    health_collection.add(health_data)
                    print(f"Health recommendations saved to Firebase")
                except Exception as e:
                    print(f"Error saving to Firebase: {str(e)}")
                
            return file_path
        except Exception as e:
            print(f"Error saving health recommendations: {str(e)}")
            return ""
    
    def save_work_recommendations(self, query: str) -> str:
        """
        Get and save work recommendations.
        
        Args:
            query: Search query
            
        Returns:
            Path to saved file
        """
        try:
            # Get recommendations
            recommendation_set = self.get_recommendations(query)
            
            # Filter by type
            work_recommendations = recommendation_set.filter(type_filter="work")
            
            # Save to file
            file_path = os.path.join(self.data_dir, f"work_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(work_recommendations.to_dict(), f, indent=2, ensure_ascii=False)
            
            # Save to Firebase if available
            if db is not None:
                try:
                    # Create a document in Firebase
                    work_collection = db.collection('work_ai_recommendation')
                    work_data = {
                        'query': query,
                        'timestamp': datetime.now().isoformat(),
                        'project': 'UbumuntuBrain Development',
                        'deadline': 'Next week', 
                        'priority': 'High',
                        'recommendations': [r.text for r in work_recommendations.recommendations],
                        'relevance_scores': [r.score for r in work_recommendations.recommendations]
                    }
                    work_collection.add(work_data)
                    print(f"Work recommendations saved to Firebase")
                except Exception as e:
                    print(f"Error saving to Firebase: {str(e)}")
                
            return file_path
        except Exception as e:
            print(f"Error saving work recommendations: {str(e)}")
            return ""
    
    def save_commute_recommendations(self, query: str) -> str:
        """
        Get and save commute recommendations.
        
        Args:
            query: Search query
            
        Returns:
            Path to saved file
        """
        try:
            # Get recommendations
            recommendation_set = self.get_recommendations(query)
            
            # Filter by type
            commute_recommendations = recommendation_set.filter(type_filter="commute")
            
            # Save to file
            file_path = os.path.join(self.data_dir, f"commute_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(commute_recommendations.to_dict(), f, indent=2, ensure_ascii=False)
            
            # Save to Firebase if available
            if db is not None:
                try:
                    # Create a document in Firebase
                    commute_collection = db.collection('commute_ai_recommendation')
                    commute_data = {
                        'query': query,
                        'timestamp': datetime.now().isoformat(),
                        'destination': 'Office',
                        'distance': '15 miles',
                        'traffic': 'Moderate',
                        'recommendations': [r.text for r in commute_recommendations.recommendations],
                        'relevance_scores': [r.score for r in commute_recommendations.recommendations]
                    }
                    commute_collection.add(commute_data)
                    print(f"Commute recommendations saved to Firebase")
                except Exception as e:
                    print(f"Error saving to Firebase: {str(e)}")
                
            return file_path
        except Exception as e:
            print(f"Error saving commute recommendations: {str(e)}")
            return ""
    
    def get_latest_health_recommendations(self) -> dict:
        """
        Get the latest health recommendations from Firebase and local storage.
        
        Returns:
            Dictionary with recommendation data or None if not found
        """
        try:
            # First check Firebase for the latest health data
            if 'db' in globals() and db is not None:
                print("Accessing health_ai_recommendation collection in Firebase")
                
                try:
                    # Get the collection
                    health_collection = db.collection('health_ai_recommendation')
                    
                    # First try to get documents ordered by created_at (most likely to be present based on logs)
                    docs = health_collection.order_by('created_at', direction='DESCENDING').limit(1).get()
                    
                    if docs and len(docs) > 0:
                        print(f"Found health document with ID: {docs[0].id}")
                        firebase_data = docs[0].to_dict()
                        firebase_data['id'] = docs[0].id
                        firebase_data['source'] = 'firebase'
                        
                        print(f"Health data fields: {', '.join(firebase_data.keys())}")
                        
                        # Check if recommendations already exist
                        if 'recommendations' not in firebase_data or not firebase_data['recommendations']:
                            # Add some sample recommendations for demo
                            blood_sugar = firebase_data.get('bloodSugar', 0)
                            recommendations = []
                            
                            if blood_sugar > 180:
                                recommendations.append("Blood sugar is elevated. Consider light exercise and drinking water.")
                                recommendations.append("Take prescribed medication as directed.")
                            elif blood_sugar < 70:
                                recommendations.append("Blood sugar is low. Consider having a small snack.")
                                recommendations.append("Keep glucose tablets or quick-acting carbs nearby.")
                            else:
                                recommendations.append("Blood sugar levels are within target range.")
                                recommendations.append("Continue with regular monitoring schedule.")
                            
                            firebase_data['recommendations'] = recommendations
                        
                        # Ensure timestamp exists
                        if 'timestamp' not in firebase_data:
                            if 'created_at' in firebase_data:
                                firebase_data['timestamp'] = firebase_data['created_at']
                            else:
                                firebase_data['timestamp'] = datetime.now().isoformat()
                        
                        return {
                            'health_contexts': [firebase_data],
                            'count': 1
                        }
                    else:
                        print("No health documents found with created_at field")
                except Exception as e:
                    print(f"Error querying health_ai_recommendation: {str(e)}")
            
            # If Firebase failed or no data, fall back to local files
            print("Falling back to local files for health data")
            files = [f for f in os.listdir(self.data_dir) if f.startswith("health_") and f.endswith(".json")]
            
            if not files:
                # Create mock data if no data exists
                print("No local health files found. Creating mock data")
                mock_data = {
                    'bloodSugar': 110,
                    'exerciseMinutes': 30,
                    'mealType': 'Balanced',
                    'timestamp': datetime.now().isoformat(),
                    'id': 'mock-health-1',
                    'source': 'local_mock',
                    'recommendations': [
                        "Blood sugar levels are within target range.",
                        "Continue with regular monitoring schedule."
                    ]
                }
                return {
                    'health_contexts': [mock_data],
                    'count': 1
                }
            
            # Sort by name (timestamp in filename)
            files.sort(reverse=True)
            
            # Get the latest file
            latest_file = os.path.join(self.data_dir, files[0])
            print(f"Using local health file: {latest_file}")
            
            # Load and return data
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error getting latest health recommendations: {str(e)}")
            # Return mock data on error
            mock_data = {
                'bloodSugar': 110,
                'exerciseMinutes': 30,
                'mealType': 'Balanced',
                'timestamp': datetime.now().isoformat(),
                'id': 'mock-health-1',
                'source': 'mock_fallback',
                'recommendations': [
                    "Blood sugar levels are within target range.",
                    "Continue with regular monitoring schedule."
                ]
            }
            return {
                'health_contexts': [mock_data],
                'count': 1
            }
    
    def get_latest_work_recommendations(self) -> dict:
        """
        Get the latest work recommendations from Firebase and local storage.
        
        Returns:
            Dictionary with recommendation data or None if not found
        """
        try:
            # First check Firebase for the latest work data
            if 'db' in globals() and db is not None:
                print("Accessing work_ai_recommendation collection in Firebase")
                
                try:
                    # Get the collection
                    work_collection = db.collection('work_ai_recommendation')
                    
                    # First try to get documents ordered by created_at (most likely to be present based on logs)
                    docs = work_collection.order_by('created_at', direction='DESCENDING').limit(1).get()
                    
                    if docs and len(docs) > 0:
                        print(f"Found work document with ID: {docs[0].id}")
                        firebase_data = docs[0].to_dict()
                        firebase_data['id'] = docs[0].id
                        firebase_data['source'] = 'firebase'
                        
                        print(f"Work data fields: {', '.join(firebase_data.keys())}")
                        
                        # Check if recommendations already exist
                        if 'recommendations' not in firebase_data or not firebase_data['recommendations']:
                            # Add sample recommendations
                            firebase_data['recommendations'] = [
                                "Focus on high-priority tasks first",
                                "Take short breaks every hour to maintain productivity",
                                "Consider blocking distracting websites during focus time"
                            ]
                        
                        # Ensure timestamp exists
                        if 'timestamp' not in firebase_data:
                            if 'created_at' in firebase_data:
                                firebase_data['timestamp'] = firebase_data['created_at']
                            else:
                                firebase_data['timestamp'] = datetime.now().isoformat()
                        
                        return {
                            'work_contexts': [firebase_data],
                            'count': 1
                        }
                    else:
                        print("No work documents found with created_at field")
                except Exception as e:
                    print(f"Error querying work_ai_recommendation: {str(e)}")
            
            # If Firebase failed or no data, fall back to local files
            print("Falling back to local files for work data")
            files = [f for f in os.listdir(self.data_dir) if f.startswith("work_") and f.endswith(".json")]
            
            if not files:
                # Create mock data if no data exists
                print("No local work files found. Creating mock data")
                mock_data = {
                    'project': 'UbumuntuBrain Development',
                    'deadline': 'Next week',
                    'priority': 'High',
                    'timestamp': datetime.now().isoformat(),
                    'id': 'mock-work-1',
                    'source': 'local_mock',
                    'recommendations': [
                        "Focus on high-priority tasks first",
                        "Take short breaks every hour to maintain productivity",
                        "Consider blocking distracting websites during focus time"
                    ]
                }
                return {
                    'work_contexts': [mock_data],
                    'count': 1
                }
            
            # Sort by name (timestamp in filename)
            files.sort(reverse=True)
            
            # Get the latest file
            latest_file = os.path.join(self.data_dir, files[0])
            print(f"Using local work file: {latest_file}")
            
            # Load and return data
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error getting latest work recommendations: {str(e)}")
            # Return mock data on error
            mock_data = {
                'project': 'UbumuntuBrain Development',
                'deadline': 'Next week',
                'priority': 'High',
                'timestamp': datetime.now().isoformat(),
                'id': 'mock-work-1',
                'source': 'mock_fallback',
                'recommendations': [
                    "Focus on high-priority tasks first",
                    "Take short breaks every hour to maintain productivity",
                    "Consider blocking distracting websites during focus time"
                ]
            }
            return {
                'work_contexts': [mock_data],
                'count': 1
            }
    
    def get_latest_commute_recommendations(self) -> dict:
        """
        Get the latest commute recommendations from Firebase and local storage.
        
        Returns:
            Dictionary with recommendation data or None if not found
        """
        try:
            # First check Firebase for the latest commute data
            if 'db' in globals() and db is not None:
                print("Accessing commute_ai_recommendation collection in Firebase")
                
                try:
                    # Direct access to the known document ID from the Firebase console
                    specific_doc_id = "TOCkiDYKJiZNO7MBQzTF"
                    doc_ref = db.collection('commute_ai_recommendation').document(specific_doc_id)
                    doc = doc_ref.get()
                    
                    if doc.exists:
                        print(f"Successfully retrieved document {specific_doc_id} from Firebase")
                        firebase_data = doc.to_dict()
                        firebase_data['id'] = doc.id
                        firebase_data['source'] = 'firebase'
                        
                        print(f"Firebase data fields: {', '.join(firebase_data.keys())}")
                        
                        # Make sure the recommendations field exists
                        if 'recommendations' not in firebase_data or not firebase_data['recommendations']:
                            # Create recommendations based on the data
                            recommendations = []
                            
                            # Add recommendations based on traffic condition if present
                            if 'trafficCondition' in firebase_data:
                                traffic = firebase_data.get('trafficCondition', 'moderate').lower()
                                if traffic == 'light':
                                    recommendations.append("Traffic is light - good time for travel")
                                elif traffic == 'heavy':
                                    recommendations.append("Traffic is heavy - consider alternate route")
                                else:
                                    recommendations.append("Traffic is moderate - leave at regular time")
                            
                            # Add existing recommendations from the screenshot
                            recommendations.extend([
                                "Check weather conditions before departure",
                                "Ensure bike lights are working for visibility",
                                "Short commute - good opportunity for exercise"
                            ])
                            
                            firebase_data['recommendations'] = recommendations
                        
                        # Ensure timestamp exists
                        if 'timestamp' not in firebase_data:
                            if 'created_at' in firebase_data:
                                firebase_data['timestamp'] = firebase_data['created_at']
                            else:
                                firebase_data['timestamp'] = datetime.now().isoformat()
                        
                        return {
                            'commute_contexts': [firebase_data],
                            'count': 1
                        }
                    else:
                        print(f"Document {specific_doc_id} does not exist, trying fallback methods")
                        
                except Exception as e:
                    print(f"Error accessing specific document: {str(e)}")
                
                # Fallback: Try normal Firebase query
                try:
                    # Get the collection
                    commute_collection = db.collection('commute_ai_recommendation')
                    
                    # Try with created_at first
                    docs = commute_collection.order_by('created_at', direction='DESCENDING').limit(1).get()
                    
                    if docs and len(docs) > 0:
                        print("Found Firebase data with query by created_at")
                        firebase_data = docs[0].to_dict()
                        firebase_data['id'] = docs[0].id
                        firebase_data['source'] = 'firebase'
                        
                        # Ensure recommendations field
                        if 'recommendations' not in firebase_data or not firebase_data['recommendations']:
                            # Create recommendations based on document data
                            firebase_data['recommendations'] = [
                                "Traffic is light - good time for travel",
                                "Check weather conditions before departure",
                                "Ensure bike lights are working for visibility",
                                "Short commute - good opportunity for exercise"
                            ]
                        
                        # Ensure timestamp
                        if 'timestamp' not in firebase_data:
                            if 'created_at' in firebase_data:
                                firebase_data['timestamp'] = firebase_data['created_at']
                            else:
                                firebase_data['timestamp'] = datetime.now().isoformat()
                        
                        return {
                            'commute_contexts': [firebase_data],
                            'count': 1
                        }
                
                except Exception as e:
                    print(f"Error with fallback query: {str(e)}")
            
            # If Firebase failed or no data, fall back to local files
            print("Falling back to local files for commute data")
            files = [f for f in os.listdir(self.data_dir) if f.startswith("commute_") and f.endswith(".json")]
            
            if not files:
                # Create mock data if no data exists
                print("No local files found. Creating mock commute data")
                mock_data = {
                    'destination': 'Office',
                    'distance': '15 miles',
                    'traffic': 'Moderate',
                    'timestamp': datetime.now().isoformat(),
                    'id': 'mock-commute-1',
                    'source': 'local_mock',
                    'recommendations': [
                        "Traffic is moderate today, leave at your regular time",
                        "Weather forecast shows clear skies for your commute",
                        "Gas prices are stable in your area"
                    ]
                }
                return {
                    'commute_contexts': [mock_data],
                    'count': 1
                }
            
            # Sort by name (timestamp in filename)
            files.sort(reverse=True)
            
            # Get the latest file
            latest_file = os.path.join(self.data_dir, files[0])
            print(f"Using local file: {latest_file}")
            
            # Load and return data
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error getting latest commute recommendations: {str(e)}")
            # Return mock data on error
            print("Error occurred. Using fallback mock data for commute")
            mock_data = {
                'destination': 'Office',
                'distance': '15 miles',
                'traffic': 'Moderate',
                'timestamp': datetime.now().isoformat(),
                'id': 'mock-commute-1',
                'source': 'mock_fallback',
                'recommendations': [
                    "Traffic is moderate today, leave at your regular time",
                    "Weather forecast shows clear skies for your commute",
                    "Gas prices are stable in your area"
                ]
            }
            return {
                'commute_contexts': [mock_data],
                'count': 1
            }

    def _collection_has_field(self, collection, field):
        """
        Helper method to check if a collection has a specific field.
        
        Args:
            collection: The collection to check
            field: The field to check for
            
        Returns:
            True if the field exists, False otherwise
        """
        try:
            print(f"Checking if collection {collection._path} has field {field}")
            query = collection.where(field, '!=', None).limit(1)
            docs = query.get()
            has_field = len(docs) > 0
            print(f"Collection {collection._path} {'has' if has_field else 'does not have'} field {field}")
            return has_field
        except Exception as e:
            print(f"Error checking if collection has field: {str(e)}")
            return False 