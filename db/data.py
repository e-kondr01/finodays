from db import conn


def get_train_data():
    cursor = conn.cursor()
    cursor.execute("""SELECT mood, text_msg FROM training_table""")
    return cursor.fetchall()


def get_test_data():
    cursor = conn.cursor()
    cursor.execute("""SELECT mood, text_msg FROM test_table""")
    return cursor.fetchall()
