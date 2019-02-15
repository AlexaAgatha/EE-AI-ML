import numpy as np
import matplotlib.pyplot as plt


F = np.array([[0, 0], [0, 1]])
E = np.array([-8, 0])

# Eqn of Parabola is AtFA + EA = 0
# Eqn of tangent must be AtFA1 + 0.5E(A+A1) = 0 (SS1=0)[Given, it passes through [-8, 0]]
	# => GA = 0


Threshold = 0.02 #Define a small threshold to get an approximate solution

l = np.linspace(-10, 10, 10000)
sol1 = 0

for x in l:
	D = np.array([8, x])
	K = np.matmul(D, F)
	L = np.matmul(K, D)
	M = np.matmul(E, D)
	N= np.abs(L + M)
	if N < Threshold:
		sol1 = x


l = np.linspace(10, -10, 10000)
sol2 = 0

for x in l:
	D = np.array([8, x])
	K = np.matmul(D, F)
	L = np.matmul(K, D)
	M = np.matmul(E, D)
	N= np.abs(L + M)
	if N < Threshold:
		sol2 = x


print(sol1)
print(sol2)

A1 = np.array([8, sol1])
A2 = np.array([8, sol2])

Area = 0.5*np.linalg.det([[A1[0], A1[1], 1],[A2[0], A2[1], 1], [2, 0, 1] ])

print(np.abs(Area))


p = np.linspace(0, 15, 10000)
plt.plot(p, np.sqrt(8*p), color = 'blue', label = 'Given Parabola')
plt.plot(p, -np.sqrt(8*p), color = 'blue')

plt.plot([A1[0], -8], [A1[1], 0], color = 'black', label = 'Tangents')
plt.plot([A2[0], -8], [A2[1], 0], color = 'black')

Focus = np.array([2, 0])

plt.plot([A2[0], Focus[0]], [A2[1], Focus[1]], color = 'red')
plt.plot([A1[0], Focus[0]], [A1[1], Focus[1]], color = 'red')
plt.plot([A2[0], A1[0]], [A2[1], A1[1]], color = 'red', label = 'Area Required to Calculate')

plt.text(-8 - 0.5, 0 + 0.5 , "0")
plt.text(A1[0] - 0.5, A1[1] + 0.5 , "A(8, 8)")
plt.text(A2[0] - 0.5, A2[1] + 0.5 , "B(8, -8)")
plt.text(2 - 0.5, 1 + 0.5 , "F(2, 0)")

plt.plot(A1[0], A1[1], "o", color = 'black')
plt.plot(A2[0], A2[1], "o", color = 'black')
plt.plot(-8, 0, "o", color = 'black')
plt.plot(2, 0, "o", color = 'black')


plt.legend()
plt.grid()

plt.savefig('/home/gautham/Desktop/AIML/Fig.png')

plt.show()









