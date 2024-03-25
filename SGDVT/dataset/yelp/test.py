import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

#
# df = pd.read_csv("./interaction.txt",sep=' ')
#
# df_grouped_u = df.groupby('user').agg('count')
# df_grouped_u = df_grouped_u.reset_index(drop=False)
# df_grouped_u.rename(columns={'item': 'inter_n'}, inplace=True)
#
#
# df_inter_u = df_grouped_u.groupby('inter_n').agg('count')
# df_inter_u = df_inter_u.reset_index(drop=False)
# df_inter_u .rename(columns={'user':'user_n'}, inplace=True)
# print(df_inter_u)
#
# plt.bar(df_inter_u['inter_n'],df_inter_u['user_n'])
# plt.title('inter_count_statistics')
# plt.xlabel('inter_n')
# plt.ylabel('user_n')
# plt.show()
#
#
#
# df_grouped_i = df.groupby('item').agg('count')
# df_grouped_i = df_grouped_i.reset_index(drop=False)
# df_grouped_i.rename(columns={'user': 'intered_n'}, inplace=True)
#
#
# df_inter_i = df_grouped_i.groupby('intered_n').agg('count')
# df_inter_i = df_inter_i.reset_index(drop=False)
# df_inter_i .rename(columns={'item':'item_n'}, inplace=True)
# print(df_inter_i)
#
#
# plt.bar(df_inter_i['intered_n'],df_inter_i['item_n'])
# plt.title('inter_count_statistics')
# plt.xlabel('intered_n')
# plt.ylabel('item_n')
# plt.show()

# def csr_matrix_average(matrix):
#     num_nonzero = matrix.count_nonzero()
#     if num_nonzero == 0:
#         return 0
#
#     total_sum = np.sum(matrix.data)
#     average = total_sum / num_nonzero
#
#     return average
#
# def zero_out_diagonal(matrix):
#     if not isinstance(matrix, csr_matrix):
#         raise ValueError("Input matrix must be a csr_matrix.")
#
#     num_rows = matrix.shape[0]
#
#     for i in range(num_rows):
#         row_start = matrix.indptr[i]
#         row_end = matrix.indptr[i+1]
#
#         for j in range(row_start, row_end):
#             if matrix.indices[j] == i:
#                 matrix.data[j] = 0
#
#     return matrix
#
# InterNet = pd.read_csv("./interaction.txt",sep=' ')
# user_n =len(set(InterNet['user']))
# item_n =len(set(InterNet['item']))
#
# UINet = csr_matrix((np.ones(len(InterNet)), (InterNet['user'], InterNet['item'])),
#                                     shape=(user_n, item_n ))
# U_I_UNet = UINet.dot(UINet.T)
# U_I_UNet = zero_out_diagonal(U_I_UNet)
#
# UI_avg = csr_matrix_average(U_I_UNet)
# print("UI_avg")
# print(UI_avg)
#
# friendNet = pd.read_csv("./trust.txt",sep=' ')
# socialNet = csr_matrix((np.ones(len(friendNet)), (friendNet['user'], friendNet['friend'])),
#                                     shape=(user_n, user_n))
# socialNet = socialNet + socialNet.T.multiply(socialNet.T >socialNet)-socialNet.multiply(socialNet.T >socialNet)
#
# U_U = socialNet.multiply(U_I_UNet)
# social_avg = csr_matrix_average(U_U)
# print("social_avg")
# print(social_avg)
#
# inter_ratio = social_avg/UI_avg
# print("inter_ratio")
# print(inter_ratio)

InterNet = pd.read_csv("interaction.txt", sep=' ')
user_n =len(set(InterNet['user']))
item_n =len(set(InterNet['item']))


print("用户总数")
print(user_n)
print("物品总数")
print(item_n)
rating_density = len(InterNet)/user_n/item_n
print("rating_density")
print(rating_density)


UINet = csr_matrix((np.ones(len(InterNet)), (InterNet['user'], InterNet['item'])),
                                    shape=(user_n, item_n ))
U_I_UNet = UINet.dot(UINet.T)
Valid_Ratio = U_I_UNet.nnz/(user_n*user_n)
print("Valid_Ratio")
print(Valid_Ratio)

friendNet = pd.read_csv("trust.txt", sep=' ')
socialNet = csr_matrix((np.ones(len(friendNet)), (friendNet['user'], friendNet['friend'])),
                                    shape=(user_n, user_n))
socialNet = socialNet + socialNet.T.multiply(socialNet.T >socialNet)-socialNet.multiply(socialNet.T >socialNet)
U_U = socialNet.multiply(U_I_UNet)
Valid_socail_Ratio = U_U.nnz/socialNet.nnz
print("Valid_socail_Ratio")
print(Valid_socail_Ratio)

social_diffuse_level = Valid_socail_Ratio/Valid_Ratio
print("social_diffuse_level")
print(social_diffuse_level)
