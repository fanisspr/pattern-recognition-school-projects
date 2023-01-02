import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity as kd

# class Parzen_window: 
#     def __init__(self, data: np.ndarray) -> None:
#         self.data = data

#     #Parzen window density estimation, with gaussian kernel
#     def estimate(self, bandwidth: float, kernel: str ='gaussian') -> kd:
#         pwde = kd(kernel=kernel, bandwidth=bandwidth).fit(self.data.reshape(-1,1))
#         return pwde

#     # def evaluate(self, pwde: kd, points_num: int) -> np.ndarray:
#     #     points = np.linspace(-10, 10, points_num)
#     #     pdf_values = np.exp(pwde.score_samples(points.reshape(-1,1)))
#     #     return pdf_values

#     #Evaluate the log density model on the points.
#     def evaluate(self, points_num: int, bandwidth: float) -> np.ndarray:
#         pwde = self.estimate(bandwidth=bandwidth)
#         points = np.linspace(-10, 10, points_num)
#         pdf_values = np.exp(pwde.score_samples(points.reshape(-1,1)))
#         return pdf_values
        
#     def plot(self, points_num: int, bandwidth: float, pdf_values: np.ndarray) -> None:
#         points = np.linspace(-10, 10, points_num)
#         plt.figure()
#         plt.title("N= " + str(points_num)+ " , h= " +str(bandwidth))
#         plt.xlabel("x")
#         plt.ylabel("pdf")
#         plt.plot(points, pdf_values, 'r.')
#         plt.show()
#         # plt.savefig(f'images/Parzen')

def Parzen_window(data, N, h):
    #Parcen window density estimation
    pwde = kd(kernel='gaussian', bandwidth = h).fit(data.reshape(-1,1))


    points = np.linspace(-10,10,N)
    #Evaluate the log density model on the data.
    pdf_vals = np.exp(pwde.score_samples(points.reshape(-1,1)))

    plt.figure()
    plt.title("N= " + str(N)+ " , h= " +str(h))
    plt.xlabel("x")
    plt.ylabel("pdf")
    plt.plot(points,pdf_vals,'r.')
    plt.show()


def Knn_estim(data, N, k):
    radius = []
    points = np.linspace(-10, 10, N)
    for p in points:
        distance = data - p
        distance = np.abs(distance)
        # print("dist from "+str(p)+": ", distance)
        distance = np.sort(distance)
        # print("sorted: ",distance)
        radius.append(distance[k])

    p = []
    for r in radius:
        # For 1D data Knn density estimator estimates the density by:
        est = k / N * 0.5 / r
        p.append(est)

    points = np.linspace(-10, 10, N)
    plt.figure()
    plt.title("N= " + str(N) + " , k= " + str(k))
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.plot(points, p)
    plt.show()
    # plt.savefig('knn_')

    return p

if __name__ == '__main__':

    N = [32, 256, 5000]
    h = [0.05, 0.2]
    # fig, ax = plt.subplots(2, 3)
    # fig.tight_layout(pad=3)
    # for i, bandwidth in enumerate(h):
    #     for points in N:
    for i in h:
        for j in N:
            # distribution: p=0.5 , 0<x<2
            data = np.random.uniform(0, 2, j)
            Parzen_window(data,j,i)
            # data = np.random.uniform(0, 2, points)
            # pw = Parzen_window(data)
            # pdf_values = pw.evaluate(points, bandwidth)
            # pw.plot(points, bandwidth, pdf_values)


    k = [32, 64, 256]
    # distribution: p=0.5 , 0<x<2
    data = np.random.uniform(0, 2, 5000)
    for i in k:
        Knn_estim(data, 5000, i)
