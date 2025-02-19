# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:19:36 2024

@author: huangke
"""

def var_verify2(data,var_list):
#    data=xfd_data.copy()
    data=data.fillna(-9999)
    data111=pd.DataFrame({'grade':'test', '申请笔数':-1, '申请占比':-1, '样本数':-1, '坏客户数':-1, '坏客户占比':-1, 'var':'test'},index=[0])
    for i in var_list:
        print(i)
        #i='sub_score1'
        #非缺失数据
        df1=data[data[i]!=-9999]
        #缺失数据
        df2=data[data[i]==-9999]       
        df1['grade']=pd.qcut(df1[i],20,duplicates='drop')
        df1.grade = df1.grade.astype(str)
        grade_list=list(df1['grade'].unique())
        cut_list_m = [float(i.split(',')[0].replace('(','')) for i in grade_list] 
        cut_list_m.append(9999)
        cut_list_m.sort()
        label=list(range(1,len(cut_list_m),1))
        df1['label'] = pd.cut(df1[i],bins=cut_list_m,labels=label)
        #申请笔数
        df1a=df1.groupby(['grade','label'])['apply_no'].count().to_frame(name='申请笔数').reset_index()
        df1a=df1a[df1a['申请笔数']>0].sort_values(by='label').drop(columns='label')                               
        #申请占比
        df1a['申请占比']=round(df1a['申请笔数']/data.shape[0],3)
        #样本数
        df1a1=df1[df1['flag']>=0].groupby('grade')['apply_no'].count().to_frame(name='样本数').reset_index()
        df1a=pd.merge(df1a,df1a1,on=('grade'),how='left')
        #坏客户数
        df1a2=df1[df1['flag']==1].groupby('grade')['apply_no'].count().to_frame(name='坏客户数').reset_index()
        df1a=pd.merge(df1a,df1a2,on=('grade'),how='left')
        #坏客户占比
        df1a['坏客户占比']=round(df1a['坏客户数']/df1a['样本数'],3)       
        #缺失数据部分
        df2['grade']='缺失'
        #申请笔数
        df2a=df2.groupby(['grade'])['apply_no'].count().to_frame(name='申请笔数').reset_index()                              
        #申请占比
        df2a['申请占比']=round(df2a['申请笔数']/data.shape[0],3)
        #样本数
        df2a1=df2[df2['flag']>=0].groupby('grade')['apply_no'].count().to_frame(name='样本数').reset_index()
        df2a=pd.merge(df2a,df2a1,on=('grade'),how='left')
        #坏客户数
        df2a2=df2[df2['flag']==1].groupby('grade')['apply_no'].count().to_frame(name='坏客户数').reset_index()
        df2a=pd.merge(df2a,df2a2,on=('grade'),how='left')
        #坏客户占比
        df2a['坏客户占比']=round(df2a['坏客户数']/df2a['样本数'],3)
        
        #合并数据
        dfa=pd.concat([df2a,df1a],axis=0)
        dfa['var']=i
        data111=pd.concat([data111,dfa],axis=0)
    data111=data111[['var','grade', '申请笔数', '申请占比', '样本数', '坏客户数', '坏客户占比']]
    return data111


df111=var_verify2(xfd_data,var_list)
