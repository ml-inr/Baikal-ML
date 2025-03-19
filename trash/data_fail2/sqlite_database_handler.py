import sqlite3
from typing import Dict


class SQLiteDatabaseHandler:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.connection = sqlite3.connect(self.db_name)
        self.cursor = self.connection.cursor()
        self._create_tables()

    def _create_tables(self):
        # Create necessary tables in SQLite database
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY,
                data_field REAL
            )
        ''')
        self.connection.commit()

    def insert_data(self, data: Dict):
        try:
            event_ids = data.get('event_ids', [])
            data_field = data.get('data_field', [])

            for event_id, value in zip(event_ids, data_field):
                self.cursor.execute('''
                    INSERT INTO events (event_id, data_field) VALUES (?, ?)
                ''', (event_id, value))

            self.connection.commit()

        except Exception as e:
            print(f"Error inserting data into database: {str(e)}")

    def close(self):
        self.connection.close()
