import numpy as np
import gc
import json
import os
import re
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
from collections import defaultdict
from scipy.sparse import csr_matrix

import json
from sklearn.model_selection import train_test_split
def reIndex(x, Reindex):
    if str(x) in Reindex.keys():
        return Reindex[str(x)]
    else:
        return -1

def renameFriendID(friendNet: pd.DataFrame):
    with open(f"userReindex.json") as f:
        userReindex = json.load(f)
    friendNet['user'] = friendNet.apply(lambda x: reIndex(x.user, userReindex), axis=1)
    friendNet['friend'] = friendNet.apply(lambda x: reIndex(x.friend, userReindex), axis=1)
    friendNet = friendNet.drop(friendNet[(friendNet['user'] == -1) | (friendNet['friend'] == -1)].index)
    return friendNet



def filter_dataset(dataset,i):
    user_interactions = dataset['user'].value_counts()
    item_interactions = dataset['item'].value_counts()

    # 保留交互次数大于等于5的用户和物品
    valid_users = user_interactions[user_interactions > i].index
    valid_items = item_interactions[item_interactions > i].index

    # 进行筛选，删除交互次数小于5的记录
    filtered_dataset = dataset[
        (dataset['user'].isin(valid_users)) &
        (dataset['item'].isin(valid_items))
    ]

    return filtered_dataset



df = pd.read_csv('flickr.txt', sep='\t', dtype=int)
print(df)
df['rating']=1


Socialnet = pd.read_csv('social.txt', sep='\t', dtype=int)
print(Socialnet)

Socialnet = Socialnet[['user','friend']]
U_Set = set(Socialnet['user']) | set(Socialnet['friend'])


df = df[df['user'].map(lambda x:x in U_Set)]
df = df.reset_index(drop=True)

df_grouped_u = df.groupby('user').agg('count')
df_grouped_u = df_grouped_u.reset_index(drop=False)
df_grouped_u.rename(columns={'item': 'inter_n'}, inplace=True)


df_inter_u = df_grouped_u.groupby('inter_n').agg('count')
df_inter_u = df_inter_u.reset_index(drop=False)
df_inter_u .rename(columns={'user':'user_n'}, inplace=True)
print(df_inter_u)

plt.bar(df_inter_u['inter_n'],df_inter_u['user_n'])
plt.title('inter_count_statistics')
plt.xlabel('inter_n')
plt.ylabel('user_n')
plt.show()


df_grouped_i = df.groupby('item').agg('count')
df_grouped_i = df_grouped_i.reset_index(drop=False)
df_grouped_i.rename(columns={'user': 'intered_n'}, inplace=True)


df_inter_i = df_grouped_i.groupby('intered_n').agg('count')
df_inter_i = df_inter_i.reset_index(drop=False)
df_inter_i .rename(columns={'item':'item_n'}, inplace=True)
print(df_inter_i)


plt.bar(df_inter_i['intered_n'],df_inter_i['item_n'])
plt.title('inter_count_statistics')
plt.xlabel('intered_n')
plt.ylabel('item_n')
plt.show()

del_threshold = [4,4,4,4,4,4,4,4,4,4,4,4]
for i in del_threshold:
    print(i)
    filtered_dataset = filter_dataset(df,i)
    df = pd.DataFrame(filtered_dataset)

    df_grouped_u = df.groupby('user').agg('count')
    df_grouped_u = df_grouped_u.reset_index(drop=False)
    df_grouped_u.rename(columns={'item': 'inter_n'}, inplace=True)

    df_inter_u = df_grouped_u.groupby('inter_n').agg('count')
    df_inter_u = df_inter_u.reset_index(drop=False)
    df_inter_u.rename(columns={'user': 'user_n'}, inplace=True)
    print(df_inter_u)

    plt.bar(df_inter_u['inter_n'], df_inter_u['user_n'])
    plt.title('inter_count_statistics')
    plt.xlabel('inter_n')
    plt.ylabel('user_n')
    plt.show()

    df_grouped_i = df.groupby('item').agg('count')
    df_grouped_i = df_grouped_i.reset_index(drop=False)
    df_grouped_i.rename(columns={'user': 'intered_n'}, inplace=True)

    df_inter_i = df_grouped_i.groupby('intered_n').agg('count')
    df_inter_i = df_inter_i.reset_index(drop=False)
    df_inter_i.rename(columns={'item': 'item_n'}, inplace=True)
    print(df_inter_i)

    plt.bar(df_inter_i['intered_n'], df_inter_i['item_n'])
    plt.title('inter_count_statistics')
    plt.xlabel('intered_n')
    plt.ylabel('item_n')
    plt.show()


userId = pd.Categorical(df['user'])
itemId = pd.Categorical(df['item'])
df['user'] = userId.codes
df['item'] = itemId.codes

userCodeDict = {int(value): code for code, value in enumerate(userId.categories.values)}
itemCodeDict = {int(value): code for code, value in enumerate(itemId.categories.values)}

with open(f"userReindex.json", "w") as f:
    f.write(json.dumps(userCodeDict))
with open(f"itemReindex.json", "w") as f:
    f.write(json.dumps(itemCodeDict))

df.to_csv('./interaction.txt',sep=' ', index=False)
Socialnet = renameFriendID(Socialnet)
Socialnet.to_csv('./trust.txt', sep=' ',index=False, header=False)




user_n =len(set(df['user']))
item_n =len(set(df['item']))
print("用户总数")
print(user_n)
print("物品总数")
print(item_n)
rating_density = len(df)/user_n/item_n
print("rating_density")
print(rating_density)


UINet = csr_matrix((np.ones(len(df)), (df['user'], df['item'])),
                                    shape=(user_n, item_n ))
U_I_UNet = UINet.dot(UINet.T)
Valid_Ratio = U_I_UNet.nnz/(user_n*user_n)
print("Valid_Ratio")
print(Valid_Ratio)

friendNet = Socialnet
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



train_set, test_set  = train_test_split(df, test_size=0.2, random_state=2018)
train_set.to_csv(f'./train.txt', sep=' ',index=False, header=False)
test_set.to_csv(f'./test.txt', sep=' ',index=False, header=False)

# train_set.to_csv(f'./train.txt', sep=' ',index=False)
# test_set.to_csv(f'./test.txt', sep=' ',index=False)