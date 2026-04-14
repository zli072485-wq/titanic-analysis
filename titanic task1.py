import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('titanic.csv')
#查看数据
print('===前五行===')
print(df.head())
print('===数据形状===')
print(df.shape)
#第一船舱平均年龄
first_class=df[df['Pclass']==1]
first_age=first_class['Age']
print('===第一船舱平均年龄为===')
print(first_age.mean())
#按照年龄进行分组
df['AgeGroup'] = 'Adult'
df.loc[df['Age']<18,'AgeGroup']='child'
df.loc[df['Age']>60,'AgeGroup']='Elder'
print('===分组结果===')
print(df[['Age','AgeGroup']].head(10))
#按照船舱和性别进行分组之后计算存活率
result=df.groupby(['Pclass','Sex'])['Survived'].mean()
print('===分组结果===')
print(result)
#处理缺失值
print('===各列缺失值信息===')
print(df.isnull().sum())#检查缺失值
age_median=df['Age'].median()
df['Age']=df['Age'].fillna(age_median)
df=df.drop('Cabin',axis=1)
embarked_mode=df['Embarked'].mode()[0]
df['Embarked']=df['Embarked'].fillna(embarked_mode)
print('===处理后===')
print(df.isnull().sum())
#将性别转换为数字
df['Sex_Num']=df['Sex'].map({'male':0,'female':1})
#df['Sex_Num']=df['Sex'].replace({'male':0,'female':1})
print(df[['Sex','Sex_Num']].head())
#将转换后的性别，分组后的年龄联系在一起分析存活率
result1=df.groupby(['Pclass','Sex_Num','AgeGroup'])['Survived'].mean()
print(result1)
#仅对按照性别和船舱分组的结果进行可视化
survival=df.groupby(['Pclass','Sex'])['Survived'].mean().unstack()
print(survival)
survival.plot(kind='bar', color=['red', 'blue'])#bar柱状图color第一列（男人）红色，第二列蓝色
plt.title('Survival Rate by Class and Gender')#用plt加标题
plt.xlabel('Passenger Class')#plt加x轴标签
plt.ylabel('Survival Rate')#plt加y轴标签
plt.legend(['Male', 'Female'])#图例（区分不同颜色的柱子代表什么）
plt.xticks(rotation=180)#给x轴设置刻度，0表示0度
plt.savefig('survival_by_class.png')#把当前图表保存成survival_by_class.png
plt.show()#展示
