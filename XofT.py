import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as ani
import math
Color = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]


def LnFuncp(x):
    return np.log(np.exp(x) + 1)


def LnFuncm(x):
    return np.log(np.exp(x) - 1)


def xt(x0, v0, c2, c1, c0, t):
    Delta = np.sqrt(c1 ** 2 + 4 * c2 * c0)
    c = - np.log(np.abs(Delta - 2 * c2 * v0 - c1) / (Delta + 2 * c2 * v0 + c1)) / Delta
    Vmax = (-c1 + Delta) / (2 * c2)
    if v0 <= Vmax:
        x = (LnFuncp(Delta * (t + c)) - LnFuncp(Delta * c)) / c2 - (c1 + Delta) * t / 2 / c2 + x0
        v = 1 / (2 * c2) * (2 * Delta / (1 + np.exp(-Delta * t - Delta * c)) - c1 - Delta)
    else:
        x = (LnFuncm(Delta * (t + c)) - LnFuncm(Delta * c)) / c2 - (c1 + Delta) * t / 2 / c2 + x0
        v = 1 / (2 * c2) * (2 * Delta / (1 - np.exp(-Delta * t - Delta * c)) - c1 - Delta)
    return x, v


def LoopFunc(TransOrNot, EndTime, x0, v0, t0, c2, c1, c0_TB, c0_DB, PerTrans, PerDualB):
    t = t0
    x = x0
    v = v0
    Len = PerTrans + PerDualB
    while t < EndTime:
        if TransOrNot:
            Per = math.floor(t / Len) * Len + PerTrans - t
            t += Per
            c0 = c0_TB
        else:
            Per = math.ceil(t / Len) * Len - t
            t += Per
            c0 = c0_DB
        TransOrNot = not TransOrNot
        if t < EndTime:
            x, v = xt(x, v, c2, c1, c0, Per)
        else:
            x, v = xt(x, v, c2, c1, c0, Per + EndTime - t)
    if t == EndTime:
        return x, v, EndTime, TransOrNot
    else:
        return x, v, EndTime, not TransOrNot


# Function which gives the analytic expression for acceleration process
def XofT_Analytic(v0, Coverage, c2, c1, FA, TAF, DBF, IBF, IBStartTime, LenOfBoost, LenOfIBoost, T):
    x = 0
    v = v0
    t = 0
    PerTrans = (1 - Coverage) * LenOfBoost
    PerDualB = Coverage * LenOfBoost
    TransOrNot = True
    c0_TB = 9 / 250 * FA * TAF
    c0_DB = c0_TB * DBF
    c0_ExTB = IBF * c0_TB
    c0_ExDB = IBF * c0_DB
    x, v, t, TransOrNot = LoopFunc(TransOrNot, IBStartTime, x, v, t, c2, c1, c0_TB, c0_DB, PerTrans, PerDualB)
    x, v, t, TransOrNot = LoopFunc(TransOrNot, IBStartTime + LenOfIBoost, x, v, t, c2, c1, c0_ExTB, c0_ExDB, PerTrans,
                                   PerDualB)
    x, v, t, TransOrNot = LoopFunc(TransOrNot, T, x, v, t, c2, c1, c0_TB, c0_DB, PerTrans, PerDualB)
    return x, v


# Function that numerically modelling the acceleration process
def XofT_Numeric(v0, Coverage, c2, c1, FA, TAF, DBF, IBF, IBStartTime, LenOfBoost, LenOfIBoost, T):
    dt = 0.01
    t = 0
    x = 0
    v = v0
    l = 1 - Coverage
    ArrayOfX = np.array([])
    ArrayOfV = np.array([])
    ArrayOfT = np.array([])
    c0_TB = 9 / 250 * FA * TAF
    c0_DB = c0_TB * DBF
    c0_ExTB = IBF * c0_TB
    c0_ExDB = IBF * c0_DB
    while t < T:
        Nt = t / LenOfBoost
        Ini = math.floor(Nt)
        if IBStartTime <= t < IBStartTime + LenOfIBoost:
            if Ini <= Nt < Ini + l:
                a = c0_ExTB - c1 * v - c2 * v ** 2
            else:
                a = c0_ExDB - c1 * v - c2 * v ** 2
        else:
            if Ini <= Nt < Ini + l:
                a = c0_TB - c1 * v - c2 * v ** 2
            else:
                a = c0_DB - c1 * v - c2 * v ** 2
        v += a * dt
        x += v * dt
        t += dt
        ArrayOfT = np.append(ArrayOfT, t)
        ArrayOfV = np.append(ArrayOfV, v)
        ArrayOfX = np.append(ArrayOfX, x)
    return ArrayOfX, ArrayOfV, ArrayOfT


# Fixed Coffecient
c1 = 0.03

# Coffecient of rare kart
'''
KartName = ["Beat", "Marathon", "Saber", "Estoc", "Lodi-Golden", "C-Burst",
            "Plasma", "ProtoBike0707", "ProtoBike0722", "ProtoBike0804", "ProtoBike0818",
            "Boxter"]
Coverage = np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.7])
DBF = np.array([1.07, 1.07, 1.07, 1.07, 1.07, 1.07, 1.07, 1.07, 1.065, 1.065, 1.065, 1.07])
IBF = np.array([1.14, 1.14, 1.18, 1.15, 1.18, 1.29, 1.135, 1.19, 1.18, 1.18, 1.18, 1.29])
LenofFullIBoost = np.array([2.5, 2.55, 2, 2.7, 2.2, 1.02, 3, 1.5 * 1.5, 1.5 * 1.5, 1.2 * 1.5, 1.2 * 1.5, 1.04 * 1.1])
TAF = np.array([1.859, 1.859, 1.859, 1.859, 1.8596, 1.862, 1.8588, 1.8588, 1.8622, 1.8622, 1.8622, 1.86]) - 0.003
FA = np.array([154, 154, 154, 154, 156, 154, 154, 154, 154, 154, 154, 154]) + 2350
LenOfBoost = np.array([2.97, 3.06, 3.01, 3.01, 3.01, 2.99, 2.94, 2.84, 2.84, 2.84, 2.84, 3.03])
DF = 0.7621 - np.array([0.084, 0.084, 0.084, 0.084, 0.084, 0.084, 0.084, 0.084, 0.084, 0.084, 0.084, 0.084])
BAF = np.array([1.5, 1.5, 1.5, 1.54, 1.5, 1.5, 1.5, 1.54, 1.54, 1.54, 1.54, 1.5])
SF = np.array([1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.8, 1.65, 1.4])
'''
'''
# Coffecient of Legend kart
KartName = ["Revenge", "R8", "Cyberpunk", "ProtoBike0722", "Knight"]
Coverage = np.array([0.7, 0.7, 0.7, 0.8, 0.7])
DBF = np.array([1.07, 1.07, 1.07, 1.065, 1.07])
IBF = np.array([1.14, 1.18, 1.154, 1.18, 1.29])
LenofFullIBoost = np.array([2.5, 2, 2.75, 1.5 * 1.5, 1.04])
TAF = np.array([1.8622, 1.8622, 1.8622, 1.8622, 1.8623]) - 0.003
FA = np.array([154, 154, 154, 154, 154]) + 2350
LenOfBoost = np.array([3.065, 3.065, 3.065, 2.84, 3.065])
DF = 0.7621 - np.array([0.084, 0.084, 0.084, 0.084, 0.084])
BAF = np.array([1.54, 1.54, 1.54, 1.54, 1.54])
SF = np.array([1.4, 1.4, 1.4, 1.4, 1.42])
'''
# Coffecient of displayed kart

KartName = ["Beat", "Marathon", "Saber", "C-Burst", "Plasma"]
KartColor = ["orange", "red", "gray", "cyan", "blue"]
Coverage = np.array([0.7, 0.7, 0.7, 0.7, 0.7])
DBF = np.array([1.07, 1.07, 1.07, 1.07, 1.07])
IBF = np.array([1.14, 1.14, 1.18, 1.29, 1.135])
LenofFullIBoost = np.array([2.5, 2.55, 2, 1.02, 3])
TAF = np.array([1.859, 1.859, 1.859, 1.862, 1.8588]) - 0.003
FA = np.array([154, 154, 154, 154, 154]) + 2350
LenOfBoost = np.array([2.97, 3.06, 3.01, 2.99, 2.94])
DF = 0.7621 - np.array([0.084, 0.084, 0.084, 0.084, 0.084])
BAF = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
SF = np.array([1.4, 1.4, 1.4, 1.4, 1.4])

# Depends on whether the kart drives on the grass
# DF = DF * 2

# Coffecient controlled by player
PlayerCoff = np.array([0.5, 250, 6, 0])  # ExGauge, v0, T, st
VaryingChoice = 1
VaryingBound = np.array([230, 290])

'''
# InsBoost Strategy
# Involved in how to distribute ExGauge, what is the best triggering speed and what is best triggering time
for i in range(0, len(IBF)):
    X = np.array([])
    V = np.array([])
    X_benefit = np.array([])
    AxisOfX = np.linspace(VaryingBound[0], VaryingBound[1], 1000)
    for Variable in AxisOfX:
        PlayerCoff[VaryingChoice] = Variable
        LenOfIBoost = LenofFullIBoost[i] * PlayerCoff[0]
        x_rf, v_rf = XofT_Analytic(PlayerCoff[1], Coverage, DF[i] / 360, c1, FA[i], TAF[i], DBF,
                                   IBF[i], 0, LenOfBoost[i], 0, PlayerCoff[2])
        x, v = XofT_Analytic(PlayerCoff[1], Coverage, DF[i] / 360, c1, FA[i], TAF[i], DBF,
                             IBF[i], PlayerCoff[3], LenOfBoost[i], LenOfIBoost, PlayerCoff[2])
        X_benefit = np.append(X_benefit, (x - x_rf) / PlayerCoff[0] / PlayerCoff[2])
        V = np.append(V, v)
        X = np.append(X, x)
    if i == 0:
        X0 = X
    plt.figure(1)
    plt.plot(AxisOfX, X_benefit, label=KartName[i])
    plt.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
    plt.figure(2)
    plt.plot(AxisOfX, V, label=KartName[i])
    plt.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
    plt.figure(3)
    plt.plot(AxisOfX, X - X0, label=KartName[i])
    plt.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
'''
'''
# Basic performance of InsBoost
plt.figure(4)
x0, v0, t0 = XofT_Numeric(PlayerCoff[1], Coverage[0], DF[0] / 360, c1, FA[0], TAF[0], DBF[0],
                          IBF[0], PlayerCoff[3], LenOfBoost[0], LenofFullIBoost[0] * PlayerCoff[0], PlayerCoff[2])
for i in range(0, len(KartName)):
    x1, v1, t1 = XofT_Numeric(PlayerCoff[1], Coverage[i], DF[i] / 360, c1, FA[i], TAF[i], DBF[i],
                              IBF[i], PlayerCoff[3], LenOfBoost[i], LenofFullIBoost[i] * PlayerCoff[0], PlayerCoff[2])
    x2, v2 = XofT_Analytic(PlayerCoff[1], Coverage[i], DF[i] / 360, c1, FA[i], TAF[i], DBF[i],
                           IBF[i], PlayerCoff[3], LenOfBoost[i], LenofFullIBoost[i] * PlayerCoff[0], PlayerCoff[2])
    print(x1[-1], x2)
    print(v1[-1], v2)
    if i != 3:
        plt.plot(t1, x1 - x0, label=KartName[i])
    plt.xlabel("Time / Second")
    plt.ylabel("Distance superior to the first kart")
plt.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
plt.show()
'''
# Dynamic illustrating
fig = plt.figure(6)
xo, vo, to = XofT_Numeric(0, Coverage[0], DF[0] / 360, c1, FA[0], 1, 1, IBF[0],
                          0, LenOfBoost[0], 0, PlayerCoff[2])
x0, v0, t0 = XofT_Numeric(PlayerCoff[1], Coverage[0], DF[0] / 360, c1, FA[0], TAF[0], DBF[0], IBF[0],
                          PlayerCoff[3], LenOfBoost[0], LenofFullIBoost[0] * PlayerCoff[0], PlayerCoff[2])
x = np.zeros([len(x0), len(KartName)])
v = np.zeros([len(x0), len(KartName)])
t = np.zeros([len(x0), len(KartName)])
DxMax = np.zeros(len(KartName))
DxMin = np.zeros(len(KartName))
for i in range(0, len(KartName)):
    x1, v1, t1 = XofT_Numeric(PlayerCoff[1], Coverage[i], DF[i] / 360, c1, FA[i], TAF[i], DBF[i], IBF[i],
                              PlayerCoff[3], LenOfBoost[i], LenofFullIBoost[i] * PlayerCoff[0], PlayerCoff[2])
    x2, v2 = XofT_Analytic(PlayerCoff[1], Coverage[i], DF[i] / 360, c1, FA[i], TAF[i], DBF[i], IBF[i],
                           PlayerCoff[3], LenOfBoost[i], LenofFullIBoost[i] * PlayerCoff[0], PlayerCoff[2])
    x[:, i], v[:, i], t[:, i] = x1, v1, t1
    DxMax[i] = np.max(x1)
    DxMin[i] = np.min(x1)
    # print(x[-1, i], x2)
    # print(v[-1, i], v2)

plt.xlim(-0.3, PlayerCoff[2] + 0.3)
plt.axis(color='white')
ax = plt.gca()
ax.set_facecolor('black')
ax.spines['right'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['bottom'].set_color('white')
plt.tick_params(axis='x', color='white', labelcolor='white')
plt.tick_params(axis='y', color='white', labelcolor='white')
rect = fig.patch
rect.set_facecolor('black')

plt.ion()
ts = [0, 0]
xs = [0, 0]
vs = [0, 0]
'''
plt.xlabel("Time / Second", color='white')
plt.ylabel("Distance", color='white')
plt.ylim(np.min(DxMin) - 0.5, np.max(DxMax) + 0.5)
plt.pause(5)
for i in range(len(x0) - 1):
    for j in range(len(KartName)):
        ts = t[i: i + 2, j]
        xs = x[i: i + 2, j]
        if i == 0:
            plt.plot(ts, xs, color=KartColor[j], label=KartName[j])
        else:
            plt.plot(ts, xs, color=KartColor[j])
    if i == 0:
        plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
        plt.pause(3)
    plt.pause(0.01)
plt.pause(5)
'''

# Speed vs Time
plt.xlabel("Time / Second", color='white')
plt.ylabel("Velocity / km/h", color='white')
plt.ylim(-5, np.max(vo) + 20)
plt.pause(5)
for i in range(len(vo) - 1):
    ts = to[i: i+2]
    vs = vo[i: i+2]
    plt.plot(ts, vs, color='white')
    plt.pause(0.01)
plt.pause(5)

'''
# Jump platform
# Same time comparing speed
plt.figure(1)
for i in range(0, len(KartName)):
    PlayerCoff[3] = PlayerCoff[2] - LenofFullIBoost[i] * PlayerCoff[0]
    x1, v1, t1 = XofT_Numeric(PlayerCoff[1], Coverage, DF[i] / 360, c1, FA[i], TAF[i], DBF,
                              IBF[i], PlayerCoff[3], LenOfBoost[i], LenofFullIBoost[i] * PlayerCoff[0], PlayerCoff[2])
    x2, v2 = XofT_Analytic(PlayerCoff[1], Coverage, DF[i] / 360, c1, FA[i], TAF[i], DBF,
                           IBF[i], PlayerCoff[3], LenOfBoost[i], LenofFullIBoost[i] * PlayerCoff[0], PlayerCoff[2])
    if i == 1 or i == 2 or i == 6:  # Only compare Marathon, Saber and Burst
        plt.plot(t1, v1, label=KartName[i])
        print(x1[-1], v1[-1])
plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))
plt.show()
'''
'''
# Shoot VS Dual Boost
ReactTime = 0.1
ObserveTime = ReactTime + 0.5
Time = np.linspace(0, ObserveTime, 1000)
InitialSpeed = 250
KartIndex = 4
StandardF = FA[KartIndex] * TAF[KartIndex] * DBF[KartIndex]
NewPosition, NewSpeed = xt(0, InitialSpeed, DF[KartIndex], c1, FA[KartIndex], ReactTime)
x1_M = np.array([])
x2_M = np.array([])

plt.figure(5)
for t in Time:
    if t < ReactTime:
        Position = 0
        Speed = InitialSpeed
        Force = FA[KartIndex]
        DuringT = t
    else:
        Position = NewPosition
        Speed = NewSpeed
        Force = FA[KartIndex] * BAF[KartIndex] * SF[KartIndex]
        DuringT = t - ReactTime
    x1, v1 = xt(Position, Speed, DF[KartIndex], c1, Force, DuringT)
    x2, v2 = xt(0, InitialSpeed, DF[KartIndex], c1, StandardF, t)
    x1_M = np.append(x1_M, x1)
    x2_M = np.append(x2_M, x2)
    plt.xlabel("Time / Second")
    plt.ylabel("Distance")

plt.plot(Time, x1_M, label="Shoot")
plt.plot(Time, x2_M, label="DualBoost")
plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))
plt.show()
'''
