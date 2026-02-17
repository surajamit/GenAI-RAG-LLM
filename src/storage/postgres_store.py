import psycopg2

class PostgresStore:
    def __init__(self, dsn):
        self.conn = psycopg2.connect(dsn)

    def insert_document(self, doc_id, metadata):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (doc_id, metadata)
                VALUES (%s, %s)
                """,
                (doc_id, metadata)
            )
        self.conn.commit()
