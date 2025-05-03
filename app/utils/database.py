import sqlite3
import json
from datetime import datetime

def init_db():
    """Initialize SQLite database."""
    conn = sqlite3.connect('local_data.db')
    c = conn.cursor()
    
    # Create tables for different types of data
    c.execute('''CREATE TABLE IF NOT EXISTS queries
                 (id TEXT PRIMARY KEY, query TEXT, timestamp TEXT, 
                  source TEXT, processed BOOLEAN, answer TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS health_context
                 (id TEXT PRIMARY KEY, blood_sugar TEXT, created_at TEXT,
                  exercise_minutes TEXT, meal_type TEXT, medication TEXT,
                  notes TEXT, timestamp TEXT, type TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS work_context
                 (id TEXT PRIMARY KEY, task_name TEXT, status TEXT,
                  priority TEXT, collaborators TEXT, deadline TEXT,
                  notes TEXT, timestamp TEXT, type TEXT, created_at TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS commute_context
                 (id TEXT PRIMARY KEY, duration TEXT, end_location TEXT,
                  notes TEXT, start_location TEXT, timestamp TEXT,
                  traffic_condition TEXT, transport_mode TEXT, type TEXT,
                  created_at TEXT)''')
    
    conn.commit()
    conn.close()

def save_to_local_db(data_type, data):
    """Save data to local SQLite database."""
    conn = sqlite3.connect('local_data.db')
    c = conn.cursor()
    
    try:
        # Convert Firebase timestamps to strings
        if isinstance(data.get('timestamp'), datetime):
            data['timestamp'] = data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(data.get('created_at'), datetime):
            data['created_at'] = data['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            
        if data_type == 'query':
            c.execute('''INSERT OR REPLACE INTO queries 
                        (id, query, timestamp, source, processed, answer)
                        VALUES (?, ?, ?, ?, ?, ?)''',
                     (str(data.get('id')), str(data.get('query')), str(data.get('timestamp')),
                      str(data.get('source')), bool(data.get('processed', False)),
                      json.dumps(data.get('answer', {}))))
        
        elif data_type == 'health_context':
            c.execute('''INSERT OR REPLACE INTO health_context 
                        (id, blood_sugar, created_at, exercise_minutes,
                         meal_type, medication, notes, timestamp, type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (str(data.get('id')), str(data.get('bloodSugar')), str(data.get('created_at')),
                      str(data.get('exerciseMinutes')), str(data.get('mealType')),
                      str(data.get('medication')), str(data.get('notes', '')), str(data.get('timestamp')),
                      str(data.get('type'))))
        
        elif data_type == 'work_context':
            c.execute('''INSERT OR REPLACE INTO work_context 
                        (id, task_name, status, priority, collaborators,
                         deadline, notes, timestamp, type, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (str(data.get('id')), str(data.get('taskName')), str(data.get('status')),
                      str(data.get('priority')), str(data.get('collaborators')),
                      str(data.get('deadline')), str(data.get('notes', '')), str(data.get('timestamp')),
                      str(data.get('type')), str(data.get('created_at'))))
        
        elif data_type == 'commute_context':
            c.execute('''INSERT OR REPLACE INTO commute_context 
                        (id, duration, end_location, notes, start_location,
                         timestamp, traffic_condition, transport_mode, type, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (str(data.get('id')), str(data.get('duration')), str(data.get('endLocation')),
                      str(data.get('notes', '')), str(data.get('startLocation')), str(data.get('timestamp')),
                      str(data.get('trafficCondition')), str(data.get('transportMode')),
                      str(data.get('type')), str(data.get('created_at'))))
        
        conn.commit()
    except Exception as e:
        print(f"Error saving to local database: {str(e)}")
    finally:
        conn.close()

def get_from_local_db(data_type, limit=1):
    """Retrieve data from local SQLite database."""
    conn = sqlite3.connect('local_data.db')
    c = conn.cursor()
    
    try:
        if data_type == 'query':
            c.execute('SELECT * FROM queries ORDER BY timestamp DESC LIMIT ?', (limit,))
        elif data_type == 'health_context':
            c.execute('SELECT * FROM health_context ORDER BY created_at DESC LIMIT ?', (limit,))
        elif data_type == 'work_context':
            c.execute('SELECT * FROM work_context ORDER BY created_at DESC LIMIT ?', (limit,))
        elif data_type == 'commute_context':
            c.execute('SELECT * FROM commute_context ORDER BY created_at DESC LIMIT ?', (limit,))
        
        rows = c.fetchall()
        if not rows:
            return None
        
        # Convert row to dictionary
        columns = [description[0] for description in c.description]
        data = dict(zip(columns, rows[0]))
        
        # Convert JSON strings back to objects
        if data_type == 'query' and data.get('answer'):
            data['answer'] = json.loads(data['answer'])
        
        return data
    except Exception as e:
        print(f"Error retrieving from local database: {str(e)}")
        return None
    finally:
        conn.close() 