import sqlite3
import pandas as pd
import numpy as np
import zlib

class LocalSQLiteDatabase:
    def __init__(self, db_name='local_database.db'):
        self.db_name = db_name
        self.connection = sqlite3.connect(self.db_name)
        self.cursor = self.connection.cursor()
        # Setting PRAGMAs for performance improvement during bulk operations
        self.cursor.execute("PRAGMA journal_mode = OFF;")
        self.cursor.execute("PRAGMA synchronous = OFF;")
        self.cursor.execute("PRAGMA cache_size = 100000;")
        print(f"Connected to SQLite database: {self.db_name}")

    def create_table(self, table_name, schema):
        try:
            self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({schema});")
            self.connection.commit()
            print(f"Table {table_name} created successfully.")
        except Exception as e:
            print(f"Error creating table {table_name}: {e}")

    def store_dataframe(self, dataframe, table_name, if_exists='replace'):
        try:
            # Optimize DataFrame for storage
            optimized_df = self.optimize_dataframe(dataframe)

            # Create table if it does not exist
            if if_exists == 'replace':
                columns = ", ".join([f"{col} {self.get_sqlite_type(optimized_df[col])}" for col in optimized_df.columns])
                self.create_table(table_name, columns)

            # Use executemany() for inserting data in chunks
            placeholders = ", ".join(["?"] * len(optimized_df.columns))
            insert_query = f"INSERT INTO {table_name} VALUES ({placeholders})"

            # Convert DataFrame to a list of tuples for executemany()
            data_tuples = [tuple(row) for row in optimized_df.to_numpy()]

            self.cursor.executemany(insert_query, data_tuples)
            self.connection.commit()

            print(f"DataFrame stored successfully in table {table_name}.")
        except Exception as e:
            print(f"Error storing DataFrame to table {table_name}: {e}")

    def optimize_dataframe(self, dataframe):
        """Optimize dataframe to reduce its size."""
        optimized_df = dataframe.copy()

        # Convert object types with low cardinality to categories
        for col in optimized_df.select_dtypes(include=['object']).columns:
            if optimized_df[col].nunique() / len(optimized_df) < 0.5:
                optimized_df[col] = optimized_df[col].astype('category')

        # Downcast numerical columns to save memory
        for col in optimized_df.select_dtypes(include=['int']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')

        for col in optimized_df.select_dtypes(include=['float']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')

        # Apply compression for text-heavy columns (e.g., comments, reviews)
        for col in optimized_df.select_dtypes(include=['category']).columns:
            optimized_df[col] = optimized_df[col].apply(lambda x: zlib.compress(x.encode('utf-8')) if pd.notnull(x) else x)
        print("DataFrame has been optimized!")
        return optimized_df

    def get_sqlite_type(self, series):
        """Infer SQLite data type from pandas Series"""
        if pd.api.types.is_integer_dtype(series):
            return "INTEGER"
        elif pd.api.types.is_float_dtype(series):
            return "REAL"
        elif pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
            return "BLOB"
        else:
            return "TEXT"

    def retrieve_dataframe(self, query):
        try:
            dataframe = pd.read_sql_query(query, self.connection)

            # Decompress text columns if they were compressed
            for col in dataframe.columns:
                if dataframe[col].dtype == 'object':
                    try:
                        dataframe[col] = dataframe[col].apply(lambda x: zlib.decompress(x).decode('utf-8') if isinstance(x, bytes) else x)
                    except:
                        pass  # Skip if decompression fails (e.g., non-compressed column)

            print("DataFrame retrieved successfully.")
            return dataframe
        except Exception as e:
            print(f"Error retrieving DataFrame: {e}")

    def close_connection(self):
        if self.connection:
            self.connection.close()
            print("SQLite connection closed.")

# Example usage:
if __name__ == '__main__':
    # Create an instance of the local SQLite database handler
    db = LocalSQLiteDatabase(db_name='my_local_database.db')

    # Create a new table called "example_table"
    # Example schema: (name TEXT, age INTEGER)
    db.create_table('example_table', 'name BLOB, age INTEGER')

    # Create a sample DataFrame with mixed types
    data = {'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35]}
    df = pd.DataFrame(data)

    # Store DataFrame in the SQLite database
    db.store_dataframe(df, 'example_table')

    # Retrieve data from the SQLite database
    retrieved_df = db.retrieve_dataframe('SELECT * FROM example_table')
    print(retrieved_df)

    # Close the database connection
    db.close_connection()
