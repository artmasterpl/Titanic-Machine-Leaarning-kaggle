import pandas
import matplotlib.pyplot as plt

df = pandas.read_csv("train.csv")
fig = plt.figure(figsize=(18, 6))

plt.subplot2grid((2, 3), (0, 0))
df.Survived.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Survived")

plt.subplot2grid((2, 3), (0, 1))
plt.scatter(df.Sex=="male", df.Age, alpha=0.1)
plt.title("Age wrt Survived")

plt.subplot2grid((2, 3), (0, 2))
df.Pclass.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Class")

plt.subplot2grid((2, 3), (1, 0), colspan=1)
for x in [1, 2, 3]:
    df.Age[df.Pclass == x].plot(kind="kde")
plt.title("Class wrt Age")
plt.legend(("1st", "2nd", "3rd"))

plt.subplot2grid((2, 3), (1, 2))
df.Embarked.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Embarked")

plt.subplot2grid((2, 3), (1, 1))
df.Sex.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Sex")
plt.show()