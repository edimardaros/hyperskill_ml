import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plt.close()

for i in range(10):
    plt.figure(i)

plt.close(1)
plt.show()