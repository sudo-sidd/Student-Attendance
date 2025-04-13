import sqlite3

conn = sqlite3.connect('attendance.db')

cur = conn.cursor()

queries = 