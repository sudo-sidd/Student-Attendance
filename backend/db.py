# import sqlite3
# from datetime import datetime

# # Connect to SQLite database
# conn = sqlite3.connect("attendance2.db")
# cursor = conn.cursor()

# # Function to create tables
# def create_tables():
#     cursor.executescript("""
#         CREATE TABLE IF NOT EXISTS Departments (
#             dept_id INTEGER PRIMARY KEY AUTOINCREMENT,
#             dept_name TEXT NOT NULL UNIQUE
#         );
#         CREATE TABLE IF NOT EXISTS Batches (
#             batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
#             dept_id INTEGER NOT NULL,
#             year INTEGER NOT NULL,
#             FOREIGN KEY (dept_id) REFERENCES Departments(dept_id)
#         );
#         CREATE TABLE IF NOT EXISTS Sections (
#             section_id INTEGER PRIMARY KEY AUTOINCREMENT,
#             batch_id INTEGER NOT NULL,
#             section_name TEXT NOT NULL,
#             FOREIGN KEY (batch_id) REFERENCES Batches(batch_id)
#         );
#         CREATE TABLE IF NOT EXISTS Students (
#             student_id INTEGER PRIMARY KEY AUTOINCREMENT,
#             register_number TEXT NOT NULL UNIQUE,
#             name TEXT NOT NULL,
#             section_id INTEGER NOT NULL,
#             FOREIGN KEY (section_id) REFERENCES Sections(section_id)
#         );
#         CREATE TABLE IF NOT EXISTS Subjects (
#             subject_id INTEGER PRIMARY KEY AUTOINCREMENT,
#             subject_code TEXT NOT NULL UNIQUE,
#             subject_name TEXT NOT NULL,
#             dept_id INTEGER NOT NULL,
#             year INTEGER NOT NULL,
#             FOREIGN KEY (dept_id) REFERENCES Departments(dept_id)
#         );
#         CREATE TABLE IF NOT EXISTS Timetable (
#             timetable_id INTEGER PRIMARY KEY AUTOINCREMENT,
#             section_id INTEGER NOT NULL,
#             day_of_week TEXT NOT NULL,
#             period_number INTEGER NOT NULL,
#             start_time TEXT NOT NULL,
#             end_time TEXT NOT NULL,
#             subject_id INTEGER NOT NULL,
#             FOREIGN KEY (section_id) REFERENCES Sections(section_id),
#             FOREIGN KEY (subject_id) REFERENCES Subjects(subject_id)
#         );
#         CREATE TABLE IF NOT EXISTS Attendance (
#             attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
#             student_id INTEGER NOT NULL,
#             timetable_id INTEGER NOT NULL,
#             date TEXT NOT NULL,
#             is_present INTEGER NOT NULL DEFAULT 0,
#             FOREIGN KEY (student_id) REFERENCES Students(student_id),
#             FOREIGN KEY (timetable_id) REFERENCES Timetable(timetable_id),
#             UNIQUE(student_id, timetable_id, date)
#         );
#     """)
#     conn.commit()

# # Function to populate initial data
# def populate_database():
#     # Insert Departments
#     departments = [("CSE",), ("ECE",), ("AIDS",)]
#     cursor.executemany("INSERT INTO Departments (dept_name) VALUES (?)", departments)

#     # Get department IDs
#     cursor.execute("SELECT dept_id, dept_name FROM Departments")
#     dept_map = {name: id for id, name in cursor.fetchall()}

#     # Define structure for batches and sections
#     structure = {
#         "CSE": {1: ["A", "B", "C"], 2: ["A", "B", "C"], 3: ["A", "B", "C"], 4: ["A", "B", "C"]},
#         "ECE": {1: ["A", "B"], 2: ["A", "B"], 3: ["A"], 4: ["A"]},
#         "AIDS": {1: ["A", "B", "C"], 2: ["A", "B", "C"], 3: ["A", "B"], 4: ["A", "B"]}
#     }

#     # Insert Batches and Sections
#     batch_data = []
#     section_data = []
#     batch_id = 1
#     section_id = 1

#     for dept_name, years in structure.items():
#         dept_id = dept_map[dept_name]
#         for year, sections in years.items():
#             batch_data.append((dept_id, year))
#             current_batch_id = batch_id
#             batch_id += 1
#             for section_name in sections:
#                 section_data.append((current_batch_id, section_name))
#                 section_id += 1

#     cursor.executemany("INSERT INTO Batches (dept_id, year) VALUES (?, ?)", batch_data)
#     cursor.executemany("INSERT INTO Sections (batch_id, section_name) VALUES (?, ?)", section_data)

#     # Get section and batch mappings
#     cursor.execute("SELECT section_id, batch_id, section_name FROM Sections")
#     section_map = {(row[1], row[2]): row[0] for row in cursor.fetchall()}
#     cursor.execute("SELECT batch_id, dept_id, year FROM Batches")
#     batch_map = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

#     # Insert Dummy Students
#     students = []
#     for batch_id, (dept_id, year) in batch_map.items():
#         dept_name = [k for k, v in dept_map.items() if v == dept_id][0]
#         sections = structure[dept_name][year]
#         for section_name in sections:
#             section_id = section_map[(batch_id, section_name)]
#             for i in range(1, 4 + (section_id % 2)):
#                 reg_num = f"{dept_name}{year}{section_name}{str(i).zfill(3)}"
#                 name = f"Student {dept_name} {year}{section_name}{i}"
#                 students.append((reg_num, name, section_id))

#     cursor.executemany("INSERT INTO Students (register_number, name, section_id) VALUES (?, ?, ?)", students)

#     # Insert Subjects (corrected to match 4 columns)
#     subjects = [
#         ("CS101", "Programming Fundamentals", dept_map["CSE"], 1),
#         ("CS102", "Digital Logic", dept_map["CSE"], 1),
#         ("CS103", "Mathematics I", dept_map["CSE"], 1),
#         ("CS201", "Data Structures", dept_map["CSE"], 2),
#         ("CS202", "Database Systems", dept_map["CSE"], 2),
#         ("CS203", "Algorithms", dept_map["CSE"], 2),
#         ("EC101", "Basic Electronics", dept_map["ECE"], 1),
#         ("EC102", "Circuit Theory", dept_map["ECE"], 1),
#         ("EC103", "Mathematics I", dept_map["ECE"], 1),
#         ("AI101", "Introduction to AI", dept_map["AIDS"], 1),
#         ("AI102", "Statistics", dept_map["AIDS"], 1),
#         ("AI103", "Programming Basics", dept_map["AIDS"], 1),
#     ]
#     cursor.executemany("INSERT INTO Subjects (subject_code, subject_name, dept_id, year) VALUES (?, ?, ?, ?)", subjects)

#     # Insert Timetable (example for CSE 1st Year, Section A)
#     cursor.execute("SELECT section_id FROM Sections WHERE batch_id = 1 AND section_name = 'A'")
#     section_id = cursor.fetchone()[0]
#     cursor.execute("SELECT subject_id FROM Subjects WHERE subject_code = 'CS101'")
#     subject_id = cursor.fetchone()[0]
#     timetable = [
#         (section_id, "Thursday", 1, "08:30", "09:15", subject_id)
#     ]
#     cursor.executemany("INSERT INTO Timetable (section_id, day_of_week, period_number, start_time, end_time, subject_id) VALUES (?, ?, ?, ?, ?, ?)", timetable)

#     conn.commit()
#     print("Database populated successfully!")

# # Function to mark attendance (unchanged)
# def mark_attendance(section_id, detected_register_numbers, date="2025-04-10"):
#     cursor.execute("""
#         SELECT timetable_id 
#         FROM Timetable 
#         WHERE section_id = ? 
#         AND day_of_week = 'Thursday' 
#         AND start_time = '08:30' 
#         AND end_time = '09:15'
#     """, (section_id,))
#     timetable_result = cursor.fetchone()
#     if not timetable_result:
#         print("Error: No timetable slot found for 8:30 - 9:15.")
#         return
#     timetable_id = timetable_result[0]

#     cursor.execute("SELECT student_id, register_number FROM Students WHERE section_id = ?", (section_id,))
#     all_students = cursor.fetchall()

#     attendance_data = []
#     for student_id, reg_num in all_students:
#         is_present = 1 if reg_num in detected_register_numbers else 0
#         attendance_data.append((student_id, timetable_id, date, is_present))

#     cursor.executemany("""
#         INSERT OR REPLACE INTO Attendance (student_id, timetable_id, date, is_present) 
#         VALUES (?, ?, ?, ?)
#     """, attendance_data)

#     conn.commit()
#     print(f"Attendance marked for {date}, 8:30 - 9:15.")

# # Run the script
# if __name__ == "__main__":
#     # create_tables()
#     # populate_database()

#     # Mark attendance for CSE 1st Year, Section A
#     cursor.execute("SELECT section_id FROM Sections WHERE batch_id = 25 AND section_name = 'A'")
#     section_id = cursor.fetchone()[0]
#     detected_students = ["AIDS1A001", "AIDS1A003"]
#     mark_attendance(section_id, detected_students, "2025-04-10")

#     # Verify attendance
#     cursor.execute("""
#         SELECT s.register_number, s.name, a.is_present 
#         FROM Attendance a 
#         JOIN Students s ON a.student_id = s.student_id 
#         JOIN Timetable t ON a.timetable_id = t.timetable_id 
#         WHERE t.section_id = ? AND t.start_time = '08:30' AND a.date = '2025-04-10'
#     """, (section_id,))
#     print("\nAttendance for CSE 1st Year, Section A (8:30 - 9:15):")
#     for row in cursor.fetchall():
#         print(row)

#     conn.close()

import sqlite3
from datetime import datetime

# Connect to SQLite database
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

# Function to mark attendance (from previous snippet)
def mark_attendance(section_id, detected_register_numbers, date="2025-04-10"):
    cursor.execute("""
        SELECT timetable_id 
        FROM Timetable 
        WHERE section_id = ? 
        AND day_of_week = 'Thursday' 
        AND start_time = '08:30' 
        AND end_time = '09:15'
    """, (section_id,))
    timetable_result = cursor.fetchone()
    if not timetable_result:
        print("Error: No timetable slot found for 8:30 - 9:15.")
        return
    timetable_id = timetable_result[0]

    cursor.execute("SELECT student_id, register_number FROM Students WHERE section_id = ?", (section_id,))
    all_students = cursor.fetchall()

    attendance_data = []
    for student_id, reg_num in all_students:
        is_present = 1 if reg_num in detected_register_numbers else 0
        attendance_data.append((student_id, timetable_id, date, is_present))

    cursor.executemany("""
        INSERT OR REPLACE INTO Attendance (student_id, timetable_id, date, is_present) 
        VALUES (?, ?, ?, ?)
    """, attendance_data)

    conn.commit()
    print(f"Attendance marked for {date}, 8:30 - 9:15.")

# Assuming tables and data are already populated (from previous snippets)

# Mark attendance for CSE 1st Year, Section A
cursor.execute("SELECT section_id FROM Sections WHERE batch_id = 1 AND section_name = 'A'")
section_id_result = cursor.fetchone()
if section_id_result:
    section_id = section_id_result[0]
    detected_students = ["CSE1A001", "CSE1A003"]  # Corrected to match CSE students
    mark_attendance(section_id, detected_students, "2025-04-10")

    # Verify attendance
    cursor.execute("""
        SELECT s.register_number, s.name, a.is_present 
        FROM Attendance a 
        JOIN Students s ON a.student_id = s.student_id 
        JOIN Timetable t ON a.timetable_id = t.timetable_id 
        WHERE t.section_id = ? AND t.start_time = '08:30' AND a.date = '2025-04-10'
    """, (section_id,))
    print("\nAttendance for CSE 1st Year, Section A (8:30 - 9:15):")
    for row in cursor.fetchall():
        print(row)
else:
    print("Error: Section not found for CSE 1st Year, Section A.")

conn.close()