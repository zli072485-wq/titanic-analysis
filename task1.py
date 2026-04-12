import pandas as pd
#读取表前5行和数据形状（行和列）
df=pd.read_csv('titanic.csv')
print('===前五行===')
print(df.head())
print('===数据形状===')
print(df.shape)
#计算1仓乘客平均年龄
first_class=df[df['Pclass']==1]
first_age=first_class['Age']
print(first_class)
print(first_age.mean())
#新建一个名为Adult的列，按照Age进行分类
df['AgeGroup'] = 'Adult'
df.loc[df['Age']<18,'AgeGroup']='child'
df.loc[df['Age']>60,'AgeGroup']='Elder'
print(df[['Age','AgeGroup']].head(10))
#根据仓和性别进行分组，并计算各组平均值
result=df.groupby(['Pclass','Sex'])['Survived'].mean()
print(result)