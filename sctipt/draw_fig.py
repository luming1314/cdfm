import warnings
warnings.filterwarnings("ignore")


import matplotlib.pyplot as plt

# 打印所有可用的颜色映射名称
print(plt.cm.datad.keys())


"""
这个是中间2个凸起，四周逐渐降低的形状
"""

# import numpy as np
# import matplotlib.pyplot as plt
#
# # 定义两个高斯函数的参数和网格
# x0, y0 = -0.42, -0.42
# x1, y1 = 0.42, 0.42
# sigma = 0.5
# amplitude = 2
#
# x = np.linspace(-2, 2, 400)
# y = np.linspace(-2, 2, 400)
# X, Y = np.meshgrid(x, y)
#
# Z1 = amplitude * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))
# Z2 = amplitude * np.exp(-((X - x1) ** 2 + (Y - y1) ** 2) / (2 * sigma ** 2))
# Z = Z1 + Z2
#
# # 绘制填充等高线图，不显示网格和坐标轴
# custom_colors = ['white', '#CCE2E8', '#9DCBDD', '#7AAECE', '#638FB4']
# plt.figure(figsize=(8, 8))
# plt.contourf(X, Y, Z, 10, cmap='Blues')
# # plt.contourf(X, Y, Z, 4, colors=custom_colors)
# plt.gca().xaxis.set_visible(False)
# plt.gca().yaxis.set_visible(False)
# plt.grid(False)
# #
# # # 显示图形
# plt.show()
# # 保存图像，设置透明背景
# # plt.savefig('contour_plot_transparent.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
#
# # 关闭图形，因为我们不需要显示它
# plt.close()




"""

这个是中间凸起1个，周围下降的 标准圆环图
"""

# #
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 定义高斯函数的参数
# x0, y0 = 0, 0  # 高斯函数的中心
# sigma = 1.0  # 标准差，控制凸起的宽度
# amplitude = 1.0  # 振幅，控制凸起的高度
#
# # 创建x和y的网格
# x = np.linspace(-3 * sigma, 3 * sigma, 400)
# y = np.linspace(-3 * sigma, 3 * sigma, 400)
#
# X, Y = np.meshgrid(x, y)
#
# # 计算二维高斯函数
# Z = amplitude * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))
#
# # 绘制等高线图
# plt.figure(figsize=(8, 8))
# contour_plot = plt.contourf(X, Y, Z, cmap='Oranges', levels=8)  # 使用20个等高线级别和'viridis'颜色映射
#
# # 添加颜色条（如果需要的话）
# # plt.colorbar(contour_plot)
#
# # 隐藏坐标轴
# plt.gca().xaxis.set_visible(False)
# plt.gca().yaxis.set_visible(False)
#
# # 设置背景为透明
# plt.gca().set_axis_off()
# plt.gca().set_facecolor('none')
#
# # 设置标题和坐标轴标签（如果需要的话）
# # plt.title('Gaussian Bump Contour Plot')
#
# # 保存图片到文件，背景透明
# # plt.savefig('gaussian_bump_contour.png', transparent=True)
#
# # 显示图形（如果需要的话）
# plt.show()
#
# # 关闭图形对象，释放内存
# plt.close()


"""
这个是我参考B站的例子
"""

# import matplotlib. pyplot as plt
# import numpy as np
# import matplotlib.tri as tri
#
# np. random.seed (19680801)
# x = np.random.uniform(-2, 2, 200)
# y = np.random.uniform(-2, 2, 200)
# z = x*np.exp(-x**2-y**2)
#
# f = plt.figure(dpi=200)
# ax1 = f.subplots()
# xi = np.linspace(-2.1,2.1, 100)
# yi = np.linspace(-2.1,2.1, 100)
# Xi,Yi = np.meshgrid(xi, yi)
# triang = tri.Triangulation(x, y)
# interpolator = tri.LinearTriInterpolator (triang, z)
# zi = interpolator(Xi, Yi)
# ax1.contour(xi, yi, zi, levels=14, linewidths=0.5, colors="k")
# ax1.contourf(xi, yi, zi, levels=14, cmap="RdBu_r" )
# plt.show()



"""
这个是形态各异的，中间凸起，周围下降的图形，可以修改随机种子
"""
#
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.tri as tri
#
# # 定义高斯函数的参数
# x0, y0 = 0, 0  # 高斯函数的中心
# sigma = 1.0  # 标准差，控制凸起的宽度
# amplitude = 1.0  # 振幅，控制凸起的高度
#
# np. random.seed (8691)
# # np. random.seed (20)
# x = np.random.uniform(-3 * sigma, 3 * sigma, 400)
# y = np.random.uniform(-3 * sigma, 3 * sigma, 400)
#
# z = amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
# triang = tri.Triangulation(x, y)
# interpolator = tri.LinearTriInterpolator (triang, z)
#
# xi = np.linspace(-3 * sigma, 3 * sigma, 400)
# yi = np.linspace(-3 * sigma, 3 * sigma, 400)
#
# Xi,Yi = np.meshgrid(xi, yi)
#
# zi = interpolator(Xi, Yi)
# X,Y,Z = Xi, Yi, zi
#
#
# # 绘制等高线图
# plt.figure(figsize=(8, 6))
# contour_plot = plt.contourf(X, Y, Z, cmap='Greens', levels=10)  # 使用20个等高线级别和'viridis'颜色映射
#
# # 添加颜色条（如果需要的话）
# # plt.colorbar(contour_plot)
#
# # 隐藏坐标轴
# plt.gca().xaxis.set_visible(True)
# plt.gca().yaxis.set_visible(True)
# plt.gca().set_axis_off()
# plt.gca().set_facecolor('none')
#
# # 设置标题和坐标轴标签（如果需要的话）
# # plt.title('Gaussian Bump Contour Plot')
#
# # 保存图片到文件，背景透明
# plt.savefig('contour_plot_transparent.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
#
# # 显示图形（如果需要的话）
# # plt.show()
#
# # 关闭图形对象，释放内存
# plt.close()



"""
这是不规则的的，中间2个凸起，周围下降
"""



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# 定义两个高斯函数的参数和网格
x0, y0 = -0.5, -0.5
x1, y1 = 0.5, 0.5
sigma = 0.5
amplitude = 2

# x = np.linspace(-2 , 2 , 400)
# y = np.linspace(-2 , 2 , 400)
# X, Y = np.meshgrid(x, y)

np. random.seed (99997)
# np. random.seed (99999)
# np. random.seed (99998)
x = np.random.uniform(-2 , 2 , 400)
y = np.random.uniform(-2 , 2 , 400)
Z1 = amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
Z2 = amplitude * np.exp(-((x - x1) ** 2 + (y - y1) ** 2) / (2 * sigma ** 2))
Z = Z1 + Z2
triang = tri.Triangulation(x, y)
interpolator = tri.LinearTriInterpolator (triang, Z)

xi = np.linspace(-2 , 2 , 400)
yi = np.linspace(-2 , 2 , 400)
Xi,Yi = np.meshgrid(xi, yi)

zi = interpolator(Xi, Yi)
X,Y,Z = Xi, Yi, zi


# 绘制填充等高线图，不显示网格和坐标轴
custom_colors = ['white', '#CCE2E8', '#9DCBDD', '#7AAECE', '#638FB4']
plt.figure(figsize=(8, 8))
plt.contourf(X, Y, Z, 10, cmap='Purples')
# plt.contourf(X, Y, Z, 4, colors=custom_colors)
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.grid(False)
#
# # 显示图形
# 设置背景为透明
plt.gca().set_axis_off()
plt.gca().set_facecolor('none')
plt.show()
# 保存图像，设置透明背景
# plt.savefig('contour_plot_transparent.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
#
# 关闭图形，因为我们不需要显示它
plt.close()





