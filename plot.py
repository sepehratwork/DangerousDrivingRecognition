import matplotlib.pyplot as plt

n = [2,3,4,5,6,7,8]

rnn =   [100, 96.03, 61.14, 28.90, 60.30, 25.96, 40.72]
gru =   [100, 93.65, 65.14, 61.93, 43.82, 21.79, 63.71]
lstm =  [100, 92.86, 63.43, 39.45, 39.70, 26.28, 55.40]

plt.plot(n, rnn, color='r', label='RNN')
plt.plot(n, gru, color='g', label='GRU')
plt.plot(n, lstm, color='b', label='LSTM')
plt.xlabel("Classes")
plt.ylabel("Accuracy")
plt.title("MobileNetV3Small")
plt.legend()
plt.show()
