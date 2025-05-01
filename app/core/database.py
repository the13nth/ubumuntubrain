import sqlite3
from contextlib import contextmanager

class Database:
    def __init__(self, db_path='local_data.db'):
        self.db_path = db_path
        self.init_db()
    
    @contextmanager
    def get_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def init_db(self):
        with self.get_db() as conn:
            c = conn.cursor()
            
            # Create queries table
            c.execute('''CREATE TABLE IF NOT EXISTS queries
                        (id TEXT PRIMARY KEY, query TEXT, timestamp TEXT, 
                         source TEXT, processed BOOLEAN, answer TEXT)''')
            
            # Create health_context table
            c.execute('''CREATE TABLE IF NOT EXISTS health_context
                        (id TEXT PRIMARY KEY, blood_sugar TEXT, created_at TEXT,
                         exercise_minutes TEXT, meal_type TEXT, medication TEXT,
                         notes TEXT, timestamp TEXT, type TEXT)''')
            
            # Create work_context table
            c.execute('''CREATE TABLE IF NOT EXISTS work_context
                        (id TEXT PRIMARY KEY, task_name TEXT, status TEXT,
                         priority TEXT, collaborators TEXT, deadline TEXT,
                         notes TEXT, timestamp TEXT, type TEXT, created_at TEXT)''')
            
            # Create commute_context table
            c.execute('''CREATE TABLE IF NOT EXISTS commute_context
                        (id TEXT PRIMARY KEY, duration TEXT, end_location TEXT,
                         notes TEXT, start_location TEXT, timestamp TEXT,
                         traffic_condition TEXT, transport_mode TEXT, type TEXT,
                         created_at TEXT)''')
            
            conn.commit()
    
    def save_data(self, table, data):
        with self.get_db() as conn:
            c = conn.cursor()
            
            # Get column names for the table
            c.execute(f"PRAGMA table_info({table})")
            columns = [info[1] for info in c.fetchall()]
            
            # Create placeholders for SQL query
            placeholders = ','.join(['?' for _ in columns])
            columns_str = ','.join(columns)
            
            # Prepare values in the correct order
            values = [str(data.get(col, '')) for col in columns]
            
            # Insert or replace data
            c.execute(f'''INSERT OR REPLACE INTO {table}
                         ({columns_str}) VALUES ({placeholders})''', values)
            conn.commit()
    
    def get_data(self, table, conditions=None, limit=1):
        with self.get_db() as conn:
            c = conn.cursor()
            
            query = f'SELECT * FROM {table}'
            if conditions:
                query += f" WHERE {conditions}"
            query += ' ORDER BY created_at DESC'
            if limit:
                query += f' LIMIT {limit}'
            
            c.execute(query)
            columns = [description[0] for description in c.description]
            rows = c.fetchall()
            
            if not rows:
                return None
            
            # Convert to dictionary
            if limit == 1:
                return dict(zip(columns, rows[0]))
            return [dict(zip(columns, row)) for row in rows]

# Create global database instance
db = Database() 