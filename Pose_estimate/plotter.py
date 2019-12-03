import matplotlib.pyplot as plt
import numpy as np


##################################################################
#                   Plot charts for demonstrating                #
##################################################################

# Test accuracy lists (for 100 epochs)
simplenet = [
    1533.129, 2160.3772, 1835.3556, 1704.9974, 1659.6865, 1635.4747, 1594.5037, 1606.2072, 1589.5798, 1635.6042,
    1564.512, 1696.5193, 1559.0416, 1572.4517, 1543.5479, 1770.0776, 1505.127, 1560.0758, 1545.8206, 1563.3273,
    1420.3257, 1524.7598, 1572.1538, 1502.2135, 1454.5607, 1504.102, 1571.5575, 1477.9781, 1430.123, 1480.3726,
    1409.8477, 1456.3083, 1435.4647, 1519.5392, 1383.4272, 1404.0972, 1510.6586, 1418.8616, 1517.5201, 1423.101,
    1402.9631, 1343.7224, 1388.0895, 1380.244, 1442.3745, 1488.9203, 1431.3833, 1358.7017, 1358.1781, 1343.5157,
    1448.2578, 1368.4845, 1407.6218, 1446.5015, 1392.904, 1411.2157, 1436.4923, 1354.1376, 1356.648, 1368.127,
    1465.5107, 1357.5117, 1371.3989, 1340.9956, 1392.9716, 1314.1147, 1391.2161, 1349.3616, 1350.6637,
    1395.5359, 1446.245, 1463.6829, 1347.0187, 1374.9784, 1360.5647, 1292.5248, 1303.127, 1305.7253,
    1361.254, 1258.8268, 1288.6311, 1590.631, 1387.9679, 1417.1228, 1390.2147, 1274.9999, 1356.2528,
    1342.1803, 1350.7106, 1277.2334, 1299.7393, 1302.652, 1282.6315, 1329.2122, 1296.6396, 1318.8302,
    1289.3007, 1278.1616, 1295.232, 1315.5315, 1268.8076
]
resnet = [
    1184.1040, 1299.2939, 1272.6082, 1247.6040, 1265.3860, 1244.2612, 1264.6820, 1214.6011, 1149.6636, 1136.3450,
    1135.4932, 1115.9512, 1240.5864, 1050.7350, 1062.2108, 1091.8334, 990.9344, 1094.1284, 982.0412, 978.4062,
    987.1945, 966.3865, 1007.7279, 947.1210, 974.4529, 1046.4874, 1000.5262, 955.2307, 924.6191, 1006.7534,
    1001.7894, 931.4714, 908.0775, 904.4051, 938.8281, 911.3879, 886.0916, 1014.5488, 938.9102, 921.9583,
    931.9586, 915.0044, 911.2165, 870.0297, 880.8435, 914.2188, 927.0396, 920.2728, 927.5316, 925.0993,
    870.9587, 945.2905, 961.2165, 896.0859, 881.3632, 1060.4191, 1023.0531, 944.6945, 896.9399, 936.1666,
    918.7401, 927.8598, 943.7512, 914.6008, 951.4264, 876.6813, 930.0831, 864.1745, 917.1935, 907.9249,
    952.8934, 936.1243, 954.1965, 926.6547, 932.5767, 891.3066, 893.0549, 885.1263, 878.2966, 889.5817,
    916.5404, 907.0396, 935.2739, 940.9099, 909.5692, 918.6119, 938.0557, 894.8992, 929.9576, 905.3394,
    939.1268, 907.9641, 917.1105, 897.2797, 905.7919, 916.9789, 886.7051, 929.1196, 901.9509, 908.4753, 928.2631
]
deepnet = [
     1337.2869, 1377.4393, 1325.1058, 1303.3262, 1378.1996, 1330.2301, 1402.2252, 1293.0457, 1271.2479, 1289.2671,
     1330.8245, 1278.2726, 1276.3464, 1288.0442, 1271.9836, 1285.0984, 1274.5509, 1289.0555, 1266.5393, 1269.124,
     1268.0596, 1259.724, 1307.4008, 1268.156, 1260.225, 1265.4921, 1275.4178, 1253.3914, 1252.8381, 1244.6288,
     1276.3944, 1252.0582, 1247.2155, 1234.9618, 1239.1521, 1235.9552, 1249.1694, 1273.5785, 1242.7809, 1197.485,
     1225.7455, 1217.5126, 1226.8795, 1195.5298, 1256.3108, 1203.6381, 1191.1987, 1176.8042, 1210.5336, 1180.6995,
     1300.2705, 1201.4474, 1142.6743, 1170.3682, 1155.5652, 1134.2344, 1167.071, 1145.5908, 1147.3783, 1127.205,
     1149.2137, 1112.5961, 1109.0729, 1223.0896, 1136.8068, 1097.7261, 1113.8223, 1078.9885, 1074.7081, 1069.2648,
     1067.5763, 1077.2706, 1071.978, 1062.4442, 1067.9846, 1035.9513, 1050.2202, 1018.15784, 1058.3004, 1079.8217,
     1034.0497, 1085.7782, 1049.7002, 1123.5385, 1028.4956, 1072.5284, 1024.6132, 1060.2153, 1018.5991, 1020.6991,
     1044.9001, 1010.9059, 1002.9139, 1105.5009, 1001.8116, 1016.74634, 1001.1606, 1004.4095, 1020.41693, 1017.3967,
    985.0224
]


# Plot the learning curve
def learning_curve():
    epochs = [e+1 for e in range(len(resnet))]
    plt.hlines(0.5654, 0, len(resnet), label='Test Loss')
    plt.plot(epochs, simplenet, label='SimpleNet')
    plt.plot(epochs, deepnet, label='DeepNet')
    plt.plot(epochs, resnet, label='ResNet')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.legend()
    plt.show()
    resnet_a = np.array(resnet)
    print(resnet_a[10*5])
    print(resnet_a.min())
    print(resnet_a.argmin())


if __name__ == '__main__':
    print("----------- start -------------")

    learning_curve()

    print("------------ end --------------")
