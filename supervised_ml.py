from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# prepare the data set
hours = [1,2,3,4,5,6,7,8,9,10]
examscores = [30,40,45,55,65,70,80,85,90,95]
students = [
    [1,30],
    [2,40],
    [3,45],
    [4,55],
    [5,65],
    [6,70],
    [7,80],
    [8,85],
    [9,90],
    [10,95]
]

# seperate input features (X) and target values (y)
X = [[hour] for hour in hours]
y = examscores

# create a linear regression model
model = LinearRegression()

#train the model
model.fit(X, y)

#get user input 
study_hours = float(input("Enter the number of hours you studied: "))

# make predictions
predicted_scores = model.predict([[study_hours]])
print(f"Predicted exam score: {predicted_scores[0]:.2f}")

#plot the data and regression line
plt.scatter(hours, examscores, color='blue', label='Actual Scores')
plt.plot(hours, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Study Hours')
plt.ylabel('Exam Scores')
plt.title('Linear Regression: Study Hours vs Exam Scores')
plt.legend()
plt.show()
