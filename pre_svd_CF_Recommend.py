import numpy as np
import pandas as pd
import scipy.sparse as ss
import pickle
import scipy.io as sio


pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)
pd.set_option('display.max_colwidth',1000)
#pd.set_option('display.max_rows',30)
#
df_train=pd.read_csv('train.csv')
print('用户：',df_train['msno'].unique().shape[0])
print('歌曲：',df_train['song_id'].nunique())


#---------------------------数据预处理-----------------------------------------------
#type={'msno','song_id','source_system_tab','source_screen_name','source_type','target'}
#去掉'source_system_tab','source_screen_name','source_type'三种不相关的特征
df_train=df_train.drop(['source_system_tab','source_screen_name','source_type'],axis=1)
print('去掉无用的特征后训练集维度：',df_train.shape)


#----------------------------特征工程-------------------------------------------------
#msno song_id属于hash编码后的特征
#计算用户对每首歌曲的打分
df_user_rating=df_train[['msno','target']].groupby('msno').sum().reset_index()

df_user_rating.rename(columns={'target':'total_rating'},inplace=True)
#print(df_user_rating,)


#每首歌曲的播放比例
df_train=pd.merge(df_train,df_user_rating)
del df_user_rating
#print('用户订阅过的音乐，及总和：\n',df_train)


#删除总打分次数为0的用户（这里打分此时为0，代表着该用户在本月是第一次来
# 或者该用户上个月订阅过音乐，但是这个月该用户流失了
#通过观察，发现索引为7377417的用户订阅的音乐次数为0，所以去掉该用户
#total_rating为0的索引
index=df_train[df_train.total_rating==0].index.tolist()
#print('index=',len(index))
df_train=df_train.drop(index=index)


print(df_train.sort_values(by=['total_rating'],ascending=False))
df_train['fractional_rating_count']=df_train['target']/df_train['total_rating']

#print(df_train)

#所有的用户和item
users=df_train['msno'].unique()
n_users=len(users)
print('用户数量为：',n_users)
items=df_train['song_id'].unique()
n_items=len(items)
print('歌曲数量为：',n_items)

#计算倒排表
from collections import defaultdict
user_items=defaultdict(set)
item_users=defaultdict(set)

#用户-物品关系矩阵
user_item_scores=ss.dok_matrix((n_users,n_items))

#重新编码用户索引
users_index=dict()
#物品索引
items_index=dict()

for i,u in enumerate(users):
    users_index[u]=i
for i,j in enumerate(items):
    items_index[j]=i
n_records=df_train.shape[0]
print('n_records=',n_records)



for i in range(n_records):
    user_index_i=users_index[df_train.iloc[i]['msno']]#用户id对应的索引
    item_index_i=items_index[df_train.iloc[i]['song_id']]#歌曲id对应的索引

    #倒排表
    user_items[user_index_i].add(item_index_i)# user_index_I所对应的所有音乐
    item_users[item_index_i].add(user_index_i)#song_id的所有订阅该音乐的用户

    #分数
    score=df_train.iloc[i]['fractional_rating_count']
    user_item_scores[user_index_i,item_index_i]=score

#保存倒排表
pickle.dump(user_items,open('user_items.pkl','wb'))
pickle.dump(item_users,open('item_users.pkl','wb'))

#保存用户-物品关系矩阵,以备后用
sio.mmwrite('user_item_scores',user_item_scores)

#保存用户、物品索引表
pickle.dump(users_index,open('users_index.pkl','wb'))
pickle.dump(items_index,open('items_index.pkl','wb'))















