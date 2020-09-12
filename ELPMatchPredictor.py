import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print(bcolors.WARNING +"VALUES MUST BE FLOAT !!!!"+ bcolors.ENDC)
home = float(input(bcolors.OKBLUE +"Home team: Teams and codes in teams.cvs, just enter the team code. \n"))
away = float(input("Away team: Teams and their codes in teams.cvs, just enter the team code. \n"+ bcolors.ENDC))

def start(home,away):
    if home > 0.47 or home < 0.00:
        return True
    elif away > 0.47 or away < 0.00:
        return True
    elif away == home:
        return True
    else:
        return False
while(start(home,away)):
    print(bcolors.WARNING +"You entered an invalid code. Try again. VALUES MUST FLOAT !!!!"+ bcolors.ENDC)
    home = float(input(bcolors.OKBLUE +"Home team: Teams and codes in teams.cvs, just enter the team code. \n"))
    away = float(input("Away team: Teams and their codes in teams.cvs, just enter the team code. \n"+ bcolors.ENDC))
    start(home,away)

data = pd.read_csv('./EPL_Set.csv', sep=",")
df = pd.DataFrame(data)

df.dropna(how="any", inplace=True)
df.drop(["Date",'Div',"Season","HTR","HTHG","HTAG","FTAG","FTHG"], axis=1,inplace=True)

lh = []
for i in range(0, len(df)):
    temp = df["HomeTeam"].values[i]
    temp = temp.replace(" ", "")
    lh.append(temp)
df["HomeTeam"] = lh

la = []
for i in range(0, len(df)):
    temp = df["AwayTeam"].values[i]
    temp = temp.replace(" ", "")
    la.append(temp)
df["AwayTeam"] = la
fig= plt.figure(figsize=(10,12))

temp = df["HomeTeam"].value_counts()
temp.plot(kind="barh")
plt.savefig('test.png')
plt.show()


temp = df["AwayTeam"].value_counts()
temp.plot(kind="barh")
plt.show()

le = preprocessing.LabelEncoder()
df['HomeTeam'] = le.fit_transform(df['HomeTeam'])
df['AwayTeam'] = le.fit_transform(df['AwayTeam'])
df['FTR'] = le.fit_transform(df['FTR'])

df["HomeTeam"] = df["HomeTeam"].astype(float)
df["AwayTeam"] = df["AwayTeam"].astype(float)

for s in range(0,8740):
    df["HomeTeam"].values[s] = df["HomeTeam"].values[s]/100
    df["AwayTeam"].values[s] = df["AwayTeam"].values[s]/100

predictors = df.drop(["FTR"], axis=1)
target = df["FTR"]
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size=0.40, random_state=0)

dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_train)
acc_dtc = round(accuracy_score(y_pred, y_train) * 100, 2)
xtest_new = np.array([home, away])
xtest_new = xtest_new.reshape(1, -1)

y_pred = dtc.predict(xtest_new)
print("Sensibility: ",acc_dtc)
print("")
if(y_pred == 1):
    print("Draw",home," ",away)
elif(y_pred == 0):
    print("Away Team Wins",away)
elif (y_pred == 2):
    print("Home Team Wins",home)
