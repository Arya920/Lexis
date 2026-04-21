import mysql.connector


class DBConnector:
    def __init__(self):
        self.connection = None
        self.db_type = None

    def connect_mysql(self, host, user, password, database):
        try:
            self.connection = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database
            )
            self.db_type = "mysql"
            return {"status": "success", "message": "Connected to MySQL"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def is_connected(self):
        return self.connection is not None and self.connection.is_connected()

    def get_tables(self):
        if not self.is_connected():
            return []

        cursor = self.connection.cursor()
        cursor.execute("SHOW TABLES")
        return [row[0] for row in cursor.fetchall()]

    def run_query(self, query):
        if not self.is_connected():
            return {"error": "No DB connection"}

        cursor = self.connection.cursor(dictionary=True)

        try:
            cursor.execute(query)

            if query.strip().lower().startswith("select"):
                return cursor.fetchall()
            else:
                self.connection.commit()
                return {"message": "Query executed successfully"}

        except Exception as e:
            return {"error": str(e)}