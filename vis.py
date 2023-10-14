import pathlib
import math
import matplotlib.pyplot as plt
import numpy as np

x = []
y = []
th = []
t = []
tt = []
v = []
a = []
w = []
v0 = 0
w0 = 0
with pathlib.Path("traj.csv").open() as f:
    for line in f.readlines():
        data = [float(v) for v in line.split(",")]
        x.append(data[0])
        y.append(data[1])
        th.append(data[2])
        t.append(data[3])
total = 0.0
for i in range(len(x)):
    dt = t[i]
    total += dt
    if i < len(x) - 1:
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        dth = th[i + 1] - th[i]
        if dth < -math.pi:
            dth = -dth + 2 * math.pi
        elif dth > math.pi:
            dth = dth - 2 * math.pi
        l = math.sqrt(dx * dx + dy * dy)
        vv = l / dt
        if dx * math.cos(th[i]) + dy * math.sin(th[i]) < 0:
            vv *= -1
        ww = dth / dt
        v.append(vv)
        w.append(ww)
    if i < len(x) - 2:
        dx2 = x[i + 2] - x[i + 1]
        dy2 = y[i + 2] - y[i + 1]
        dt2 = t[i + 1]
        l2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
        vv2 = l2 / dt2
        if dx2 * math.cos(th[i + 1]) + dy2 * math.sin(th[i + 1]) < 0:
            vv2 *= -1
        aa = (vv2 - vv) * 2 / (dt + dt2)
        a.append(aa)
    tt.append(total)
print(sum(t))
fig, ax = plt.subplots(3)
ax[0].plot(x, y)
ax[0].set_title("position")
ax[0].set_xlim([-1, 4])
ax[0].set_ylim([-2.5, 2.5])
ax[1].plot(tt[:-1], v)
ax[1].set_title("velocity")
ax[2].plot(tt[:-2], a)
ax[2].set_title("acceleration")
# ax[1].plot(tt[:-1], w)
plt.legend()
plt.show()
