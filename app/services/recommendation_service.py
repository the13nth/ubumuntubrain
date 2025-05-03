from flask import jsonify
import datetime
from ..core.firebase import get_firebase_db
from ..core.database import get_db_connection

class RecommendationService:
    @staticmethod
    def save_recommendations(data, recommendation_type):
        try:
            # Save to local SQLite database
            local_data = {
                'type': recommendation_type,
                'content': data.get('content'),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO recommendations (type, content, timestamp)
                    VALUES (?, ?, ?)
                ''', (local_data['type'], local_data['content'], local_data['timestamp']))
                conn.commit()
            
            # Save to Firebase
            db = get_firebase_db()
            collection_name = f"{recommendation_type}_recommendations"
            doc_ref = db.collection(collection_name).add({
                'content': data.get('content'),
                'timestamp': datetime.datetime.now()
            })
            
            return jsonify({
                'id': doc_ref[1].id,
                'message': f'{recommendation_type.capitalize()} recommendation saved successfully'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @staticmethod
    def send_recommendation(data, recommendation_type):
        try:
            content = data.get('content')
            if not content:
                return jsonify({'error': 'Content is required'}), 400
            
            # Save to both local and Firebase
            result = RecommendationService.save_recommendations(
                {'content': content}, 
                recommendation_type
            )
            
            # Additional processing specific to recommendation type could be added here
            
            return result
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @staticmethod
    def sync_local_recommendations_with_firebase():
        try:
            db = get_firebase_db()
            
            # Get all recommendations from local database
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM recommendations ORDER BY timestamp DESC')
                recommendations = cursor.fetchall()
            
            for rec in recommendations:
                rec_type, content, timestamp = rec[1], rec[2], rec[3]
                collection_name = f"{rec_type}_recommendations"
                
                # Check if recommendation already exists in Firebase
                query = db.collection(collection_name).where(
                    'content', '==', content
                ).where(
                    'timestamp', '==', datetime.datetime.fromisoformat(timestamp)
                ).limit(1)
                
                if not list(query.stream()):
                    # Add to Firebase if not exists
                    db.collection(collection_name).add({
                        'content': content,
                        'timestamp': datetime.datetime.fromisoformat(timestamp)
                    })
            
            return jsonify({'message': 'Recommendations synced successfully'})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500 