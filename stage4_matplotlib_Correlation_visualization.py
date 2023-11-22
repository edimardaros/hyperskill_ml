import pandas as pd

exam = pd.DataFrame({
    "hours": [10, 15, 16, 18, 12, 11, 17, 12], 
    "age": [20, 22, 21, 20, 19, 20, 20, 20], 
    "exam_grade": [76, 80, 83, 80, 75, 70, 85, 78]})

# your code here
df = pd.DataFrame(exam)

# Calculate and print the correlation matrix
print(df.corr())