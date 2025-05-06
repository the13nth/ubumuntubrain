from flask import jsonify
import numpy as np
from sklearn.decomposition import PCA
from app.services.pinecone_service import pinecone_service
from app.services.firebase_service import firebase_service
import umap
from app.config.settings import Config

class VisualizationService:
    @staticmethod
    def get_embeddings_visualization():
        """Get embeddings visualization data with improved 3D visualization including processed contexts"""
        try:
            print('[Visualization] Getting index stats from Pinecone...')
            stats = pinecone_service.index.describe_index_stats()
            
            # Get total vector count from the default namespace
            total_vectors = stats.namespaces.get('', {}).get('vector_count', 0)
            print(f'[Visualization] Found {total_vectors} total vectors in index.')
            
            if total_vectors == 0:
                print('[Visualization] No vectors found in index.')
                return jsonify({
                    'points': [],
                    'metadata': {
                        'total_points': 0,
                        'context_types': {},
                        'recommendation_types': {},
                        'categories': {}
                    }
                })

            # Get all vector IDs first
            print('[Visualization] Getting all vector IDs...')
            vector_ids = pinecone_service.list_vectors()
            if not vector_ids:
                print('[Visualization] No vector IDs found.')
                return jsonify({
                    'points': [],
                    'metadata': {
                        'total_points': 0,
                        'context_types': {},
                        'recommendation_types': {},
                        'categories': {}
                    }
                })

            # Fetch vectors in batches
            print(f'[Visualization] Fetching {len(vector_ids)} vectors in batches...')
            vector_data = pinecone_service.batch_fetch(vector_ids, batch_size=10)
            
            if not vector_data or not vector_data.get('vectors'):
                print('[Visualization] No vector data could be fetched.')
                return jsonify({
                    'points': [],
                    'metadata': {
                        'total_points': 0,
                        'context_types': {},
                        'recommendation_types': {},
                        'categories': {}
                    }
                })

            # Process the fetched vectors
            vectors, documents, metadatas, ids = [], [], [], []
            for doc_id, v in vector_data['vectors'].items():
                values = getattr(v, 'values', None)
                if not values or not isinstance(values, list) or len(values) != 384:
                    print(f"[Visualization] Skipping doc_id={doc_id} due to invalid or missing embedding values.")
                    continue
                vectors.append(values)
                metadata = getattr(v, 'metadata', {})
                metadatas.append(metadata)
                documents.append(metadata.get('text', ''))
                ids.append(doc_id)

            print(f'[Visualization] Total vectors processed: {len(vectors)}')

            if not vectors:
                print('[Visualization] No vectors could be processed.')
                return jsonify({
                    'points': [],
                    'metadata': {
                        'total_points': 0,
                        'context_types': {},
                        'recommendation_types': {},
                        'categories': {}
                    }
                })

            embeddings = np.array(vectors)
            print('[Visualization] Reducing dimensions with UMAP...')
            reducer = umap.UMAP(n_components=3, random_state=42)
            reduced_embeddings = reducer.fit_transform(embeddings)

            points = []
            context_types = {}
            recommendation_types = {}

            for i, (point, doc, meta) in enumerate(zip(reduced_embeddings, documents, metadatas)):
                doc_type = meta.get('type', 'unknown')
                color = VisualizationService.get_point_color(doc_type)
                size = VisualizationService.get_point_size(doc_type)

                # Update type counts
                if 'recommendation' in doc_type:
                    recommendation_types[doc_type] = recommendation_types.get(doc_type, 0) + 1
                else:
                    context_types[doc_type] = context_types.get(doc_type, 0) + 1

                points.append({
                    'id': ids[i],
                    'x': float(point[0]),
                    'y': float(point[1]),
                    'z': float(point[2]),
                    'text': doc[:200] + '...' if len(doc) > 200 else doc,
                    'type': doc_type,
                    'color': color,
                    'size': size,
                    'metadata': meta
                })

            print(f'[Visualization] Prepared {len(points)} points for visualization.')
            return jsonify({
                'points': points,
                'metadata': {
                    'total_points': len(points),
                    'context_types': context_types,
                    'recommendation_types': recommendation_types,
                    'categories': context_types  # For frontend compatibility
                }
            })
        except Exception as e:
            print(f"Error in get_embeddings_visualization: {str(e)}")
            return jsonify({
                'points': [],
                'metadata': {
                    'total_points': 0,
                    'context_types': {},
                    'recommendation_types': {},
                    'categories': {}
                }
            })

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