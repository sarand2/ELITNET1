from time import localtime
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D

# plt.rcParams['figure.figsize'] = (10, 8)

def Step(current_X_t_minus_1, current_H_t, current_prob_estimate, C_w, C_e, measurement_vector):
    # ---------------------------Prediction step-----------------------------
    predicted_state_estimate = current_X_t_minus_1
    predicted_prob_estimate = current_prob_estimate + C_w
    # --------------------------Observation step-----------------------------
    innovation = measurement_vector - current_H_t.dot(predicted_state_estimate)
    # print((predicted_prob_estimate).shape)
    innovation_covariance = current_H_t.dot(predicted_prob_estimate).dot(
        current_H_t.T) + C_e  # KLAIDA, AISKINTIS KODEL NE C_e o paprastas noise
    # -----------------------------Update step-------------------------------
    kalman_gain = predicted_prob_estimate.dot(current_H_t.T).dot(np.linalg.inv(innovation_covariance))
    current_state_estimate = predicted_state_estimate + kalman_gain * innovation
    # We need the size of the matrix so we can make an identity matrix.
    size = current_prob_estimate.shape[0]
    # eye(n) = nxn identity matrix.
    current_prob_estimate = (np.eye(size) - kalman_gain * current_H_t) * predicted_prob_estimate
    return current_X_t_minus_1, current_H_t, current_prob_estimate, current_state_estimate


def GenerateTraffic(iterations, mean, e_t, w_t):
    # Synthetic DDoS traffic HRPI generation
    p = 3  # number parameter as in research paper
    array_Size = (iterations,)  # size of array

    Y_t_observation = np.random.normal(mean, 0.5,
                                       size=array_Size)  # 'real' observations (normal about x, deviation=0.5)

    Y_t_observation_flipped = Y_t_observation[::-1]

    H_t = Y_t_observation_flipped[0:4 - 1].reshape((1, 3))

    P_current_priori = np.identity(p)
    X_t_minus_1 = np.zeros((3, 1));

    estimate = 0  # primary estimation
    historic_values = []
    for i in range(3, 1000):
        X_t_minus_1, H_t, P_current_priori, estimate = Step(X_t_minus_1, H_t, P_current_priori, w_t[i], e_t[i],
                                                            Y_t_observation[i])  # KOL KAS VIETOJE C_e rasome 1
        historic_values.append(np.asarray(estimate).reshape(-1))
    return historic_values


# sum_rate_not_active = np.sum(normal_not_active, axis = 0)
# traffic_probability_not_active = normal_not_active / sum_rate_not_active;
# SRE_not_active = 0
# for i in range(0, len(normal_not_active)):
#    SRE_not_active += -1 *  traffic_probability_not_active[i] * np.log2(traffic_probability_not_active[i])
# print("Unactive client contribution: " , SRE_not_active)


# normal unactive clients
normal_client_count_not_active = 792000
normal_request_rate_not_active = 0.001
normal_not_active = np.full((normal_client_count_not_active,), normal_request_rate_not_active)
# Normal active clients
normal_client_count = 800
normal_request_rate = 8
normal = np.full((normal_client_count,), normal_request_rate)
# DDoS clients
ddos_client_count = 0
ddos_request_rate = 0
ddos = np.full((ddos_client_count,), ddos_request_rate)

attack_traffic = np.append(normal, ddos, axis=0)

attack_traffic = np.append(attack_traffic, normal_not_active, axis=0)

sum_rate = np.sum(attack_traffic, axis=0)

traffic_probability = attack_traffic / sum_rate;

# SRE2 = np.sum()

# print(np.sum(attack_traffic) / len(attack_traffic))
SRE = 0
for i in range(0, len(traffic_probability)):
    SRE += -1 * traffic_probability[i] * np.log2(traffic_probability[i])
print(SRE)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

iterations = 100000
meanDDoS = 3.58 # DDoS HRPI time series expectation
meanNormal = 9.26

e_t = np.random.rand(iterations)
w_t = np.random.rand(iterations)

historic_values_Normal = GenerateTraffic(iterations, meanNormal, e_t, w_t)
historic_values_DDoS = GenerateTraffic(iterations, meanDDoS, e_t, w_t)

xNormal = [ x[0] for x in historic_values_Normal]
yNormal = [ y[1] for y in historic_values_Normal]
zNormal = [ z[2] for z in historic_values_Normal]
ax.scatter(xNormal, yNormal, zNormal, c='r', marker='.')

xDDoS = [ x[0] for x in historic_values_DDoS]
yDDoS = [ y[1] for y in historic_values_DDoS]
zDDoS = [ z[2] for z in historic_values_DDoS]
ax.scatter(xDDoS, yDDoS, zDDoS, c='b', marker='^')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

training_data_classes = []
training_data_classes.extend([0]*997)
training_data_classes.extend([1]*997)
training_data = np.vstack((historic_values_Normal, historic_values_DDoS)).tolist()

# problem = svm_problem(training_data_classes, training_data)
#
# param = svm_parameter("-q") #tiesiog kad nerekautu
#
# #10-fold cross validation:
# param.cross_validation=1
# param.nr_fold=10
#
# # kernel_type : set type of kernel function (default 2)
# #2 -- radial basis function: exp(-gamma*|u-v|^2):
# param.kernel_type=rbf
#
# #perform validation
# accuracy = svm_train(problem,param)
# print(accuracy)


# disable cv
# param.cross_validation = 0
#
# # training with 70
# trainidx = int(0.7*len(classes))
# problem = svm_problem(classes[0:trainidx], data[0:trainidx])
#
# # build svm_model
# model = svm_train(problem,param)
#
# # test with 30 data
# p_lbl, p_acc, p_prob = svm_predict(classes[trainidx:], data[trainidx:], model)
# print p_acc


def DDoSTraffic(ax) :
   interval_calculations = 1
   # Synthetic DDoS traffic HRPI generation
   iterations = 1000
   p = 3 # number parameter as in research paper
   array_Size = (iterations,) # size of array
   mean = 3.58 # DDoS HRPI time series expectation
   Y_t_observation = np.random.normal(mean, 0.5, size=array_Size) # 'real' observations (normal about x, deviation=0.5)


   # White noise processes
   # e_t:
   mean = 0
   std = 1
  # e_t = [None]*interval_calculations
   #e_t[0] = np.random.normal(mean, std, size=1)
   e_t = np.random.rand(iterations)
   # Measurement noise covariance matrix:
   C_e = np.identity(p)

   # w_t:
   meanw = 0
   stdw = 1
  # w_t = [None]*interval_calculations
   #w_t[0] = np.random.normal(meanw, stdw, size=1)
   w_t = np.random.rand(iterations)

   # State noise covariance matrix:
   C_w = np.identity(p) * 0.0001 # as in research paper

    # Let's assume that we have 3 initial values

   #Initial conditions for single estimation
   #H_t = [None] * (interval_calculations+1)

   Y_t_observation_flipped = Y_t_observation[::-1]
   H_t = Y_t_observation_flipped[0:4-1].reshape((1,3))

   P_current_priori = np.identity(p)
   X_t_minus_1 = np.zeros((3, 1));

   estimate = 0
   historic_values = []
   for i in range(3, 1000):
       X_t_minus_1, H_t, P_current_priori, estimate = Step(X_t_minus_1, H_t, P_current_priori, np.random.normal(meanw, stdw, size=1), np.random.normal(mean, std, size=1), Y_t_observation[i]) # KOL KAS VIETOJE C_e rasome 1
       historic_values.append(np.asarray(estimate).reshape(-1))


   xs = [ x[0] for x in historic_values]
   ys = [ y[1] for y in historic_values]
   zs = [ z[2] for z in historic_values]
   ax.scatter(xs, ys, zs, c='b', marker='^')
   ax.set_xlabel('X Label')
   ax.set_ylabel('Y Label')
   ax.set_zlabel('Z Label')


def NormalTraffic(ax) :
   interval_calculations = 1
   # Synthetic DDoS traffic HRPI generation
   iterations = 1000
   p = 3 # number parameter as in research paper
   array_Size = (iterations,) # size of array
   mean = 9.26 # DDoS HRPI time series expectation
   Y_t_observation = np.random.normal(mean, 0.5, size=array_Size) # 'real' observations (normal about x, deviation=0.5)


   # White noise processes
   # e_t:
   mean = 0
   std = 1
   #e_t = [None]*interval_calculations
   e_t = np.random.rand(iterations)
   #e_t[0] = np.random.normal(mean, std, size=1)

   # Measurement noise covariance matrix:
   C_e = np.identity(p)

   # w_t:
   meanw = 0
   stdw = 1
  # w_t = [None]*interval_calculations
   w_t = np.random.rand(iterations)
   #w_t[0] = np.random.normal(meanw, stdw, size=1)


   # State noise covariance matrix:
   C_w = np.identity(p) * 0.0001 # as in research paper

    # Let's assume that we have 3 initial values

   #Initial conditions for single estimation
   #H_t = [None] * (interval_calculations+1)

   Y_t_observation_flipped = Y_t_observation[::-1]
   H_t = Y_t_observation_flipped[0:4-1].reshape((1,3))

   P_current_priori = np.identity(p)
   X_t_minus_1 = np.zeros((3, 1));

   estimate = 0
   historic_values = []
   for i in range(3, iterations):
       X_t_minus_1, H_t, P_current_priori, estimate = Step(X_t_minus_1, H_t, P_current_priori, np.random.normal(meanw, stdw, size=1), np.random.normal(mean, std, size=1), Y_t_observation[i]) # KOL KAS VIETOJE C_e rasome 1
       historic_values.append(np.asarray(estimate).reshape(-1))


   xs = [ x[0] for x in historic_values]
   ys = [ y[1] for y in historic_values]
   zs = [ z[2] for z in historic_values]
   ax.scatter(xs, ys, zs, c='r', marker='o')

   #plt.show()


# print(zodynas[i])


# p0 = zodynas[0]from time import localtime
# import numpy as np
# import matplotlib.pyplot as plt
# from numpy.linalg import inv
# from mpl_toolkits.mplot3d import Axes3D
#
# # plt.rcParams['figure.figsize'] = (10, 8)
#
# def Step(current_X_t_minus_1, current_H_t, current_prob_estimate, C_w, C_e, measurement_vector):
#     #---------------------------Prediction step-----------------------------
#     predicted_state_estimate = current_X_t_minus_1
#     predicted_prob_estimate = current_prob_estimate + C_w
#     #--------------------------Observation step-----------------------------
#     innovation = measurement_vector - current_H_t.dot(predicted_state_estimate)
#     #print((predicted_prob_estimate).shape)
#     innovation_covariance = current_H_t.dot(predicted_prob_estimate).dot(current_H_t.T) + C_e  # KLAIDA, AISKINTIS KODEL NE C_e o paprastas noise
#     #-----------------------------Update step-------------------------------
#     kalman_gain = predicted_prob_estimate.dot(current_H_t.T).dot(np.linalg.inv(innovation_covariance))
#     current_state_estimate = predicted_state_estimate + kalman_gain * innovation
#     # We need the size of the matrix so we can make an identity matrix.
#     size = current_prob_estimate.shape[0]
#     # eye(n) = nxn identity matrix.
#     current_prob_estimate = (np.eye(size)-kalman_gain * current_H_t)*predicted_prob_estimate
#     return current_X_t_minus_1, current_H_t, current_prob_estimate, current_state_estimate
#
#
#
# def GenerateTraffic(iterations, mean, e_t, w_t) :
#     # Synthetic DDoS traffic HRPI generation
#     p = 3 # number parameter as in research paper
#     array_Size = (iterations,) # size of array
#
#     Y_t_observation = np.random.normal(mean, 0.5, size=array_Size) # 'real' observations (normal about x, deviation=0.5)
#
#     Y_t_observation_flipped = Y_t_observation[::-1]
#
#     H_t = Y_t_observation_flipped[0:4-1].reshape((1,3))
#
#     P_current_priori = np.identity(p)
#     X_t_minus_1 = np.zeros((3, 1));
#
#     estimate = 0 # primary estimation
#     historic_values = []
#     for i in range(3, 1000):
#         X_t_minus_1, H_t, P_current_priori, estimate = Step(X_t_minus_1, H_t, P_current_priori, w_t[i], e_t[i],  Y_t_observation[i]) # KOL KAS VIETOJE C_e rasome 1
#         historic_values.append(np.asarray(estimate).reshape(-1))
#     return historic_values
#
#
#
# #sum_rate_not_active = np.sum(normal_not_active, axis = 0)
# #traffic_probability_not_active = normal_not_active / sum_rate_not_active;
# #SRE_not_active = 0
# #for i in range(0, len(normal_not_active)):
# #    SRE_not_active += -1 *  traffic_probability_not_active[i] * np.log2(traffic_probability_not_active[i])
# #print("Unactive client contribution: " , SRE_not_active)
#
#
#
# #normal unactive clients
# normal_client_count_not_active = 792000
# normal_request_rate_not_active = 0.001
# normal_not_active = np.full((normal_client_count_not_active,), normal_request_rate_not_active)
# # Normal active clients
# normal_client_count = 800
# normal_request_rate = 8
# normal = np.full((normal_client_count,), normal_request_rate)
# # DDoS clients
# ddos_client_count = 0
# ddos_request_rate = 0
# ddos = np.full((ddos_client_count,), ddos_request_rate)
#
# attack_traffic = np.append(normal, ddos, axis=0)
#
# attack_traffic = np.append(attack_traffic, normal_not_active, axis=0)
#
# sum_rate = np.sum(attack_traffic, axis = 0)
#
# traffic_probability = attack_traffic / sum_rate;
#
# #SRE2 = np.sum()
#
# #print(np.sum(attack_traffic) / len(attack_traffic))
# SRE = 0
# for i in range(0, len(traffic_probability)):
#     SRE += -1 *  traffic_probability[i] * np.log2(traffic_probability[i])
# print(SRE)
# #fig = plt.figure()
# #ax = fig.add_subplot(111, projection='3d')
#
# #iterations = 100000
# #meanDDoS = 3.58 # DDoS HRPI time series expectation
# #meanNormal = 9.26
#
# #e_t = np.random.rand(iterations)
# #w_t = np.random.rand(iterations)
#
# #historic_values_Normal = GenerateTraffic(iterations, meanNormal, e_t, w_t)
# #historic_values_DDoS = GenerateTraffic(iterations, meanDDoS, e_t, w_t)
#
# #xNormal = [ x[0] for x in historic_values_Normal]
# #yNormal = [ y[1] for y in historic_values_Normal]
# #zNormal = [ z[2] for z in historic_values_Normal]
# #ax.scatter(xNormal, yNormal, zNormal, c='r', marker='.')
#
# #xDDoS = [ x[0] for x in historic_values_DDoS]
# #yDDoS = [ y[1] for y in historic_values_DDoS]
# #zDDoS = [ z[2] for z in historic_values_DDoS]
# #ax.scatter(xDDoS, yDDoS, zDDoS, c='b', marker='^')
#
# #ax.set_xlabel('X Label')
# #ax.set_ylabel('Y Label')
# #ax.set_zlabel('Z Label')
# #plt.show()
#
# #training_data_classes = []
# #training_data_classes.extend([0]*997)
# #training_data_classes.extend([1]*997)
# #training_data = np.vstack((historic_values_Normal, historic_values_DDoS)).tolist()
#
# #problem = svm_problem(training_data_classes, training_data)
#
# #param = svm_parameter("-q") #tiesiog kad nerekautu
#
# ##10-fold cross validation:
# #param.cross_validation=1
# #param.nr_fold=10
#
# ## kernel_type : set type of kernel function (default 2)
# ##2 -- radial basis function: exp(-gamma*|u-v|^2):
# #param.kernel_type=rbf
#
# ##perform validation
# #accuracy = svm_train(problem,param)
# #print(accuracy)
#
#
# #----------------------------------------------
#
#
#
#
# ## disable cv
# #param.cross_validation = 0
#
# ## training with 70
# #trainidx = int(0.7*len(classes))
# #problem = svm_problem(classes[0:trainidx], data[0:trainidx])
#
# ## build svm_model
# #model = svm_train(problem,param)
#
# ## test with 30 data
# #p_lbl, p_acc, p_prob = svm_predict(classes[trainidx:], data[trainidx:], model)
# #print p_acc
#
#
#
#
#
# #def DDoSTraffic(ax) :
# #    interval_calculations = 1
# #    # Synthetic DDoS traffic HRPI generation
# #    iterations = 1000
# #    p = 3 # number parameter as in research paper
# #    array_Size = (iterations,) # size of array
# #    mean = 3.58 # DDoS HRPI time series expectation
# #    Y_t_observation = np.random.normal(mean, 0.5, size=array_Size) # 'real' observations (normal about x, deviation=0.5)
#
#
# #    # White noise processes
# #    # e_t:
# #    mean = 0
# #    std = 1
# #   # e_t = [None]*interval_calculations
# #    #e_t[0] = np.random.normal(mean, std, size=1)
# #    e_t = np.random.rand(iterations)
# #    # Measurement noise covariance matrix:
# #    C_e = np.identity(p)
#
# #    # w_t:
# #    meanw = 0
# #    stdw = 1
# #   # w_t = [None]*interval_calculations
# #    #w_t[0] = np.random.normal(meanw, stdw, size=1)
# #    w_t = np.random.rand(iterations)
#
# #    # State noise covariance matrix:
# #    C_w = np.identity(p) * 0.0001 # as in research paper
#
# #     # Let's assume that we have 3 initial values
#
# #    #Initial conditions for single estimation
# #    #H_t = [None] * (interval_calculations+1)
#
# #    Y_t_observation_flipped = Y_t_observation[::-1]
# #    H_t = Y_t_observation_flipped[0:4-1].reshape((1,3))
#
# #    P_current_priori = np.identity(p)
# #    X_t_minus_1 = np.zeros((3, 1));
#
# #    estimate = 0
# #    historic_values = []
# #    for i in range(3, 1000):
# #        X_t_minus_1, H_t, P_current_priori, estimate = Step(X_t_minus_1, H_t, P_current_priori, np.random.normal(meanw, stdw, size=1), np.random.normal(mean, std, size=1), Y_t_observation[i]) # KOL KAS VIETOJE C_e rasome 1
# #        historic_values.append(np.asarray(estimate).reshape(-1))
#
#
#
#
# #    xs = [ x[0] for x in historic_values]
# #    ys = [ y[1] for y in historic_values]
# #    zs = [ z[2] for z in historic_values]
# #    ax.scatter(xs, ys, zs, c='b', marker='^')
# #    ax.set_xlabel('X Label')
# #    ax.set_ylabel('Y Label')
# #    ax.set_zlabel('Z Label')
#
#
#
#
#
#
#
#
#
#
#
# #def NormalTraffic(ax) :
# #    interval_calculations = 1
# #    # Synthetic DDoS traffic HRPI generation
# #    iterations = 1000
# #    p = 3 # number parameter as in research paper
# #    array_Size = (iterations,) # size of array
# #    mean = 9.26 # DDoS HRPI time series expectation
# #    Y_t_observation = np.random.normal(mean, 0.5, size=array_Size) # 'real' observations (normal about x, deviation=0.5)
#
#
# #    # White noise processes
# #    # e_t:
# #    mean = 0
# #    std = 1
# #    #e_t = [None]*interval_calculations
# #    e_t = np.random.rand(iterations)
# #    #e_t[0] = np.random.normal(mean, std, size=1)
#
# #    # Measurement noise covariance matrix:
# #    C_e = np.identity(p)
#
# #    # w_t:
# #    meanw = 0
# #    stdw = 1
# #   # w_t = [None]*interval_calculations
# #    w_t = np.random.rand(iterations)
# #    #w_t[0] = np.random.normal(meanw, stdw, size=1)
#
#
# #    # State noise covariance matrix:
# #    C_w = np.identity(p) * 0.0001 # as in research paper
#
# #     # Let's assume that we have 3 initial values
#
# #    #Initial conditions for single estimation
# #    #H_t = [None] * (interval_calculations+1)
#
# #    Y_t_observation_flipped = Y_t_observation[::-1]
# #    H_t = Y_t_observation_flipped[0:4-1].reshape((1,3))
#
# #    P_current_priori = np.identity(p)
# #    X_t_minus_1 = np.zeros((3, 1));
#
# #    estimate = 0
# #    historic_values = []
# #    for i in range(3, iterations):
# #        X_t_minus_1, H_t, P_current_priori, estimate = Step(X_t_minus_1, H_t, P_current_priori, np.random.normal(meanw, stdw, size=1), np.random.normal(mean, std, size=1), Y_t_observation[i]) # KOL KAS VIETOJE C_e rasome 1
# #        historic_values.append(np.asarray(estimate).reshape(-1))
#
#
# #    xs = [ x[0] for x in historic_values]
# #    ys = [ y[1] for y in historic_values]
# #    zs = [ z[2] for z in historic_values]
# #    ax.scatter(xs, ys, zs, c='r', marker='o')
#
# #    #plt.show()
#
#
#
#
#
#
#     #print(zodynas[i])
#
#
# #p0 = zodynas[0]
# #p1 = zodynas[1]
# #p2 = zodynas[2]
#
# #origin = [0,0,0]
# #X, Y, Z = zip(origin,origin,origin)
# #U, V, W = zip(p0,p1,p2)
#
# #fig = plt.figure()
# #ax = fig.add_subplot(111, projection='3d')
# #ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.01)
# #plt.show()
#
#
# #plt.plot(Y_t_observation[3], estimate)
#
# # plt.plot(Y_t_observation[3], observed_y, 'ro')
#
#
#
# #X_t = [None]*interval_calculations
# #X_t[0] = np.zeros((3, 1))
#
# #P_t = [None]*interval_calculations
# #P_t[0] = np.identity(p)
#
# #for t in range(1, 2):
# #        # a priori state:
# #        X_hat_t = X_t[t-1]
# #        P_priori = P_t[t-1] + 0.0001
#
# #       #a posteriori state:
# #        K_t = P_priori.dot(H_t[t].T).dot(inv( H_t[t].dot(P_priori).dot(H_t[t].T) + 1))
# #        X_estimation = X_hat_t + K_t.dot( Y_t_observation[t + 2] - H_t[t].dot(X_hat_t) )   # WHAT IS Y_T SUPPOSED TO BE: MEEASUREMENT OR AUTOREGRESSIVE VALUE????????
# #        P_t[t] = (np.identity(p) - K_t.dot(H_t[t])).dot(P_priori)
#
#
#
#
#
# #X_hat = []
# #X_hat[0] = 0
# #P_T = np.identity(3)
# #C_e = np.identity(3)
#
#
# #print(Y_t_observation)
# p1 = zodynas[1]
# p2 = zodynas[2]

# origin = [0,0,0]
# X, Y, Z = zip(origin,origin,origin)
# U, V, W = zip(p0,p1,p2)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.01)
# plt.show()


# plt.plot(Y_t_observation[3], estimate)

# plt.plot(Y_t_observation[3], observed_y, 'ro')


# X_t = [None]*interval_calculations
# X_t[0] = np.zeros((3, 1))

# P_t = [None]*interval_calculations
# P_t[0] = np.identity(p)

# for t in range(1, 2):
#        # a priori state:
#        X_hat_t = X_t[t-1]
#        P_priori = P_t[t-1] + 0.0001

#       #a posteriori state:
#        K_t = P_priori.dot(H_t[t].T).dot(inv( H_t[t].dot(P_priori).dot(H_t[t].T) + 1))
#        X_estimation = X_hat_t + K_t.dot( Y_t_observation[t + 2] - H_t[t].dot(X_hat_t) )   # WHAT IS Y_T SUPPOSED TO BE: MEEASUREMENT OR AUTOREGRESSIVE VALUE????????
#        P_t[t] = (np.identity(p) - K_t.dot(H_t[t])).dot(P_priori)


# X_hat = []
# X_hat[0] = 0
# P_T = np.identity(3)
# C_e = np.identity(3)


# print(Y_t_observation)


