import psycopg2

conn = psycopg2.connect(dbname='maindb_ml', user='maindb_ml',
                        password='maindb_ml', host='localhost', port=5430)
