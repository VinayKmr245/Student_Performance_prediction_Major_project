# import pandas as pd

# # Sample data
# data = [["Aari","BE728212028","Male","9535866270",4,"BE","KK College","VTU","Teamviewer",80,71,4,3,5,4]]

# # Creating a Pandas dataframe
# df = pd.DataFrame(data, columns=["Name", "Roll Number", "Gender", "Phone Number", "Semester", "Course", "College", "University", "Software", "Marks 1", "Marks 2", "Marks 3", "Marks 4", "Marks 5", "Marks 6"])

# # Writing the dataframe to a CSV file
# df.to_csv("./data.csv", index=False)

import pandas as pd
import random

# Create an empty list to store the data
data = []

# Generate 100000 random entries
for i in range(100000):
    # Generate random data
    name = "Person " + str(i+1)
    roll_number = "BE" + str(random.randint(100000000, 999999999))
    gender = random.choice(["Male", "Female", "Other"])
    phone_number = str(random.randint(1000000000, 9999999999))
    semester = random.randint(1, 8)
    course = random.choice(["BE", "B.Tech", "M.Tech", "PhD"])
    college = random.choice(["KK College", "ABC College", "XYZ College"])
    # university = random.choice(["VTU", "Anna University", "JNTU"])
    university="VTU"
    software = random.choice(["Teamviewer", "Zoom", "Skype"])
    marks1 = random.randint(0, 100)
    marks2 = random.randint(0, 100)
    marks3 = random.randint(1, 5)
    marks4 = random.randint(1, 5)
    marks5 = random.randint(1, 5)
    marks6 = random.randint(1, 5)

    # Append the data to the list
    data.append([name, roll_number, gender, phone_number, semester, course, college, university, software, marks1, marks2, marks3, marks4, marks5, marks6])

# Creating a Pandas dataframe
df = pd.DataFrame(data, columns=["Name", "Roll Number", "Gender", "Phone Number", "Semester", "Course", "College", "University", "Software", "Marks 1", "Marks 2", "Marks 3", "Marks 4", "Marks 5", "Marks 6"])

# Writing the dataframe to a CSV file
df.to_csv("data.csv", index=False)