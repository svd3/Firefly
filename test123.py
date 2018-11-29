dist = []
tdist = []
probs = []
for i in range(2000):
    if i%2==0:
        fA = env.reset().view(1, -1)
        fB = env.reset().view(1, -1)
    else:
        fA, fB = fB, fA
    fB[0,0] = fA[0,0]
    rA, rB = fA[0,0], fB[0,0]
    tA, tB = abs(fA[0,1]), abs(fB[0,1])
    #d = rA-rB
    d = rA - rB
    t = tA - tB
    vA = opt_value(fA)
    vB = opt_value(fB)
    pB = 1/(1 + torch.exp(-vB + vA))
    dist.append(d)
    tdist.append(t)
    probs.append(pB)

dist = torch.stack(dist).data.numpy()
tdist = torch.stack(tdist).data.numpy()
probs = torch.stack(probs).data.numpy()

sums, v = np.histogram(tdist, bins=10, weights=probs)
counts, _ = np.histogram(tdist, bins=10)
p = sums / counts

plt.plot(np.linspace(-0.78, 0.78, 10), p)
plt.show(0)
