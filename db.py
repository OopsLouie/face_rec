import sqlite3
import numpy as np
import ast

class FaceDatabase:
    def __init__(self, db_name="face_database.db"):
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            gender TEXT NOT NULL,
            age INTEGER NOT NULL,
            phone TEXT NOT NULL,
            notes TEXT,
            embedding TEXT NOT NULL
        )
        ''')
        self.conn.commit()

    def insert_face(self, name, gender, age, phone, notes, embedding):
        embedding_str = np.array(embedding).astype(str).tolist()
        self.cursor.execute('''
        INSERT INTO faces (name, gender, age, phone, notes, embedding)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, gender, age, phone, notes, str(embedding_str)))
        self.conn.commit()

    def get_all_faces(self):
        self.cursor.execute("SELECT * FROM faces")
        return self.cursor.fetchall()

    def get_face_by_id(self, face_id):
        self.cursor.execute("SELECT * FROM faces WHERE id = ?", (face_id,))
        return self.cursor.fetchone()

    def delete_face(self, face_id):
        self.cursor.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        self.conn.commit()

    def update_face(self, face_id, name=None, gender=None, age=None, phone=None, notes=None, embedding=None):
        update_values = []
        update_query = "UPDATE faces SET"
        if name:
            update_query += " name = ?,"
            update_values.append(name)
        if gender:
            update_query += " gender = ?,"
            update_values.append(gender)
        if age:
            update_query += " age = ?,"
            update_values.append(age)
        if phone:
            update_query += " phone = ?,"
            update_values.append(phone)
        if notes:
            update_query += " notes = ?,"
            update_values.append(notes)
        if embedding is not None:
            update_query += " embedding = ?,"
            update_values.append(str(np.array(embedding).astype(str).tolist()))

        update_query = update_query.rstrip(',') + " WHERE id = ?"
        update_values.append(face_id)
        
        self.cursor.execute(update_query, tuple(update_values))
        self.conn.commit()

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    db = FaceDatabase()

    faces = db.get_all_faces()
    for face in faces:
        print(face)

    db.close()