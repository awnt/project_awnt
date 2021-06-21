#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_palette('Set2')


# In[2]:


df=pd.read_excel('The-Big-List-20210117.xlsx',sheet_name='Reviewed',usecols='A:F')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.Stars.unique()


# In[6]:


#Replace 'Unrated' with 0
df.Stars.replace(to_replace={'Unrated':0},inplace=True)
df.Stars=df.Stars.astype(float).round(1)


# # Country

# In[7]:


#Ranking average country rating 
country_s=df.groupby('Country').agg({'Country':'count','Stars':'mean'})
country_s=country_s.rename({'Country':'Total'},axis=1)
country_s.Stars=country_s.Stars.round(1)
country_s = country_s.sort_values(by =['Total','Stars'],ascending = False).reset_index()


# In[8]:


country_s.head(10)


# In[9]:


country_s.describe()


# In[10]:


sns.barplot(x='Country',y='Total',data=country_s[0:10])
plt.xticks(rotation=45)
plt.show()


# In[139]:


sns.histplot(country_s.Stars,bins=10,kde=True)
plt.title('Country vs Stars Distribution')

plt.show()


# In[140]:


country_s['Stars'].plot(kind='box')
plt.show()


# In[243]:


sns.boxplot(y='Stars',data=country_s,showmeans=True)


# In[141]:


annotations=[i for i in country_s[0:9]['Country']]
plt.figure(figsize=(8,6))
plt.scatter('Stars','Total',
           s='Total',
           data=country_s,
           c=country_s.Country.index)
plt.xlabel('Ratings')
plt.ylabel('Total')
for i,label in enumerate(annotations):
    plt.text(country_s.Stars[i],country_s.Total[i],label,color='blue', verticalalignment='bottom',  bbox=dict(facecolor='w',edgecolor = 'w',alpha=0.4))
plt.show()


# In[181]:


annotations=[i for i in country_s[0:10]['Country']]
plt.figure(figsize=(8,6))
plt.scatter('Stars','Total',
           s='Total',
           data=country_s[0:10]) 
plt.xlabel('Ratings')
plt.ylabel('Total')
for i,label in enumerate(annotations):
    plt.text(country_s.Stars[i],country_s.Total[i],label,color='blue', verticalalignment='bottom',  bbox=dict(facecolor='w',edgecolor = 'w',alpha=0.4))
plt.show()


# In[142]:


#Ranking average brand rating 
brand_s=df.groupby('Brand').agg({'Brand':'count','Stars':'mean'})
brand_s=brand_s.rename({'Brand':'Total'},axis=1)
brand_s.Stars=brand_s.Stars.round(1)
brand_s = brand_s.sort_values(by =['Total','Stars'],ascending = False).reset_index()


# In[143]:


brand_s.loc[brand_s.Total>6].sum()


# In[144]:


brand_s[1:].describe()


# In[145]:


brand_s[0:10]


# In[146]:


annotations=[i for i in brand_s[0:10]['Brand']]
plt.figure(figsize=(8,6))
plt.scatter('Stars','Total',
           s='Total',
           data=brand_s,
           c=brand_s.Brand.index)
plt.xlabel('Ratings')
plt.ylabel('Total')
for i,label in enumerate(annotations):
    plt.text(brand_s.Stars[i],brand_s.Total[i],label,color='blue', verticalalignment='bottom',  bbox=dict(facecolor='w',edgecolor = 'w',alpha=0.4))
plt.show()


# In[147]:


annotations=[i for i in brand_s[0:10]['Brand']]
plt.figure(figsize=(8,6))
plt.scatter('Stars','Total',
           s='Total',
           data=brand_s[0:10])
plt.xlabel('Ratings')
plt.ylabel('Total')
for i,label in enumerate(annotations):
    plt.text(brand_s.Stars[i],brand_s.Total[i],label,color='blue', verticalalignment='bottom',  bbox=dict(facecolor='w',edgecolor = 'w',alpha=0.4))
plt.show()


# In[152]:


sns.histplot(brand_s.Stars,bins=10,kde=True)
plt.title('Brand vs Stars Distribution')

plt.show()


# In[149]:


style_s = df.groupby('Style').agg({'Style':'count','Stars':'mean'})
style_s=style_s.rename({'Style':'Total'},axis=1).reset_index()
style_s.Stars=style_s.Stars.round(1)


# In[150]:


style_s.describe()


# In[153]:


sns.barplot(x='Style',y='Total',data=style_s.sort_values(by='Total',ascending=False))
plt.show()


# In[154]:


df['Variety']=df['Variety'].replace('[^a-zA-Z]',' ')


# # Spicy

# In[155]:


taste=['Spicy','Chili','Hot','Yum','Curry','Pepper','Mala','Spice','Ginger']
df['Spicy']=df['Variety'].apply(lambda x : 'Spicy' if sum(1 for w in x.split(' ') if w in taste)!=0 else 'NotSpicy')


# In[156]:


df.groupby('Spicy').agg({'Stars':'mean','Spicy':'count'})


# # Flavours

# In[231]:


FV={'Chicken':'Chicken',
    'Beef':['Beef','Meat','Cow'],
    'Seafood':['Shrimp','Fish','Crab','Seafood','Oyster','Lobster','Goong','Prawn'],
    'Pork':['Pork','Moo','Prok']}


# In[232]:


df['Flavour']=df['Variety'].apply(lambda x : list(k for w in x.split(' ') for k,v in FV.items() if w in v))


# In[233]:


df=df.explode('Flavour').fillna('Other')


# In[234]:


df.head()


# In[235]:


df.Flavour.value_counts()


# In[236]:


df.loc[(df.Flavour=='Seafood')& (df.Spicy=='NotSpicy')]


# In[237]:


Flavour_Other=df.loc[df.Flavour=='Other']


# In[238]:


#Ranking average country rating 
flavour_s=df.groupby('Flavour').agg({'Flavour':'count','Stars':'mean'})
flavour_s=flavour_s.rename({'Flavour':'Total'},axis=1)
flavour_s.Stars=flavour_s.Stars.round(1)
flavour_s = flavour_s.sort_values(by =['Total','Stars'],ascending = False).reset_index()


# In[225]:


flavour_s


# # Word Clound

# In[226]:


from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[227]:


word=df['Variety'].apply(lambda x:pd.value_counts(x.split(" "))).sum(axis=0)


# Meat Theme

# In[228]:


text = " ".join(desc for desc in df.Variety)
print ("There are {} words in the combination of all review.".format(len(text)))
# Create stopword list:
stopwords = set(STOPWORDS)

stopwords.update(['Flavor','Noodle','Instant','Soup','Artificial','Flavour','Noodles','Ramen','Soba','Udon','Sauce','Tonkotsu','Curry','Bowl','Kimchi','Cup','Yakisoba','Shoyu','Rice'])
# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[229]:


#del spicy
text = " ".join(desc for desc in df.Variety)
print ("There are {} words in the combination of all review.".format(len(text)))
# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(['Flavor','Noodle','Instant','Soup','Artificial','Flavour','Noodles','Ramen','Soba','Sauce','Tonkotsu','Curry','Bowl','Kimchi','Cup','Spicy','Hot','Yakisoba','Shoyu','Rice'])

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# Soup/Taste Theme

# In[239]:


text_F_Other = " ".join(desc for desc in Flavour_Other.Variety)
print ("There are {} words in the combination of all review.".format(len(text)))

# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(['flavor','noodle','Curry','Instant','big','Soup','Artificial','Taste','Bowl','Style','Soba','Flavour','Rice','Sauce','Udon','Noodles','Ramen','Chili','ramyun','Spicy','Hot','Cup','Bowl'])

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text_F_Other)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# # NMF

# In[169]:


from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


# In[170]:


temp=df['Variety']


# In[171]:


tfidf=TfidfVectorizer(stop_words=['flavor','big','flavour','ramen','noodle','noodles','top','instant','oriental','artificial','taste','bowl','Style','cup','soup','Flavour','Rice',
                                  'Sauce','Noodles','Ramen','ramyun','premium','rice','vermicelli','Cup','udon','with','new','style','yam','green','sauce'])


# In[172]:


desc=tfidf.fit_transform(temp)


# In[173]:


words = tfidf.get_feature_names()


# In[174]:


len(words)


# In[175]:


print(desc.shape)


# In[176]:


nmf = NMF(n_components=15)
nmf_features = nmf.fit_transform(desc) 


# In[177]:


components_temp = pd.DataFrame(nmf.components_,columns=words)


# In[178]:


for i in range(0,15):
    components = components_temp.iloc[i,:].nlargest()
    print("group:{}\n{}\n".format(i+1,components))
    


# ## statistics test

# In[179]:


df.head()


# In[180]:


brand_s.Stars.hist()


# In[ ]:




