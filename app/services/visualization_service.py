from flask import jsonify
import numpy as np
from sklearn.decomposition import PCA
from app import chroma_service

class VisualizationService:
    @staticmethod
    def get_embeddings_visualization():
        """Get embeddings visualization data with improved 3D visualization including processed contexts"""
        try:
            # Use the global chroma_service instance
            data = chroma_service.get_all_documents()

            # If there's no data or no embeddings, return empty visualization
            if not data or 'embeddings' not in data or not data['embeddings']:
                return jsonify({
                    "points": [],
                    "variance_explained": [0, 0, 0],
                    "metadata": {
                        "total_points": 0,
                        "categories": {},
                        "sources": {}
                    }
                })

            # Convert embeddings to numpy array
            embeddings = np.array(data['embeddings'])
            
            # Determine number of components based on data size
            n_samples = embeddings.shape[0]
            n_features = embeddings.shape[1]
            n_components = min(3, n_samples - 1, n_features)
            
            if n_components < 1:
                return jsonify({
                    "points": [],
                    "variance_explained": [0, 0, 0],
                    "metadata": {
                        "total_points": 0,
                        "categories": {},
                        "sources": {}
                    }
                })
            
            # Apply PCA to reduce dimensions
            pca = PCA(n_components=n_components)
            reduced_embeddings = pca.fit_transform(embeddings)
            
            # Create array for 3D coordinates
            coords_3d = np.zeros((reduced_embeddings.shape[0], 3))
            coords_3d[:, :n_components] = reduced_embeddings
            
            # Normalize coordinates to [-1, 1] range for better visualization
            for i in range(3):
                if np.max(np.abs(coords_3d[:, i])) > 0:
                    coords_3d[:, i] = coords_3d[:, i] / np.max(np.abs(coords_3d[:, i]))
            
            # Prepare the visualization data with enhanced metadata
            points = []
            categories = {}
            sources = {}
            
            for i in range(len(coords_3d)):
                # Get metadata for this point
                metadata = data['metadatas'][i]
                doc_type = metadata.get('type', 'unknown')
                source = metadata.get('source', 'unknown')
                
                # Update category and source counts
                categories[doc_type] = categories.get(doc_type, 0) + 1
                sources[source] = sources.get(source, 0) + 1
                
                # Create point data with enhanced information
                point_data = {
                    'x': float(coords_3d[i, 0]),
                    'y': float(coords_3d[i, 1]),
                    'z': float(coords_3d[i, 2]),
                    'text': data['documents'][i][:200] + '...' if len(data['documents'][i]) > 200 else data['documents'][i],
                    'source': source,
                    'type': doc_type,
                    'id': metadata.get('id', f'doc_{i}'),
                    'size': VisualizationService.get_point_size(doc_type),
                    'color': VisualizationService.get_point_color(doc_type)
                }
                points.append(point_data)
            
            # Prepare variance explained (pad with zeros if needed)
            variance_explained = list(pca.explained_variance_ratio_)
            while len(variance_explained) < 3:
                variance_explained.append(0.0)
            
            # Calculate additional metadata
            metadata = {
                "total_points": len(points),
                "categories": categories,
                "sources": sources,
                "dimensions": {
                    "original": n_features,
                    "reduced": n_components,
                    "variance_explained": variance_explained
                }
            }
            
            return jsonify({
                "points": points,
                "variance_explained": variance_explained,
                "metadata": metadata
            })
            
        except Exception as e:
            print(f"Error in get_embeddings_visualization: {str(e)}")
            return jsonify({
                "error": str(e),
                "points": [],
                "variance_explained": [0, 0, 0],
                "metadata": {
                    "total_points": 0,
                    "categories": {},
                    "sources": {}
                }
            }), 500

    @staticmethod
    def get_point_color(doc_type):
        """Get color for different document types with improved visibility"""
        color_map = {
            'health_context': '#00CC00',          # Brighter green
            'work_context': '#0066FF',            # Brighter blue
            'commute_context': '#FF6600',         # Brighter orange
            'health_recommendation': '#66FF66',    # Lighter bright green
            'work_recommendation': '#66B2FF',      # Lighter bright blue
            'commute_recommendation': '#FFB366',   # Lighter bright orange
            'query': '#FF0000',                   # Bright red
            'response': '#9933FF',                # Bright purple
            'unknown': '#999999'                  # Lighter gray
        }
        return color_map.get(doc_type, color_map['unknown'])

    @staticmethod
    def get_point_size(doc_type):
        """Get size for different point types"""
        size_map = {
            'health_context': 8,
            'work_context': 8,
            'commute_context': 8,
            'health_recommendation': 10,
            'work_recommendation': 10,
            'commute_recommendation': 10,
            'query': 12,
            'response': 12,
            'unknown': 6
        }
        return size_map.get(doc_type, size_map['unknown']) 