import datetime
from concurrent.futures import ProcessPoolExecutor
import psycopg2

from text_classification.preprocess import preprocess_text


def a(it: tuple[int, str]):
    conn = psycopg2.connect(dbname='maindb_ml', user='maindb_ml',
                            password='maindb_ml', host='localhost', port=5430)
    with conn.cursor() as curs:
        curs.execute(f"""UPDATE training_table SET is_processed=true, text_msg='{normalize_str_for_sql(" ".join(preprocess_text(it[1])))}' WHERE id={it[0]};""")

    conn.commit()
    conn.close()


def normalize_str_for_sql(s: str) -> str:
    """
    Normalizes str to avoid SQL Injections

    Args:
        s: str

    Returns:
        str: normalized string with replaced ' to +CHAR(39)+
    """
    return s.replace("'", "''")


def main():
    end = False
    while not end:
        conn = psycopg2.connect(dbname='maindb_ml', user='maindb_ml',
                                password='maindb_ml', host='localhost', port=5430)
        with conn.cursor() as curs:
            curs.execute("""SELECT id, text_msg FROM training_table WHERE is_processed=false;""")
            data = curs.fetchall()
            with ProcessPoolExecutor(max_workers=10) as exec:
                data1 = list(exec.map(a, data))
            print(datetime.datetime.now())

        conn.close()


if __name__ == "__main__":
    main()
