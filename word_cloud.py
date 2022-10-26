import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import sys
from PIL import Image
from wordcloud import WordCloud , STOPWORDS, ImageColorGenerator

mask = np.array(Image.open("/home/soumen/Desktop/Dr-Sarvepalli-Radhakrishnan.webp"))

df = pd.read_csv("/home/soumen/CLT_Deep_Learning/data_1.csv")
print(df)

#comment_words = ''
stopwords = set(STOPWORDS)
 
def transform_zeros(val):
    if val.all()== 0:
        return 255
    else:
        return val
maskable_image = np.ndarray((mask.shape[0],mask.shape[1]), np.int64)


text = " ".join(Name for Name in df.Name)
wordcloud = WordCloud(width = 30, height = 30,random_state=1, colormap='Set2', collocations=False, stopwords = STOPWORDS,mask=maskable_image,mode="RGBA").generate(text)
plt.figure(figsize=[20,20])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
# store to file
plt.savefig("news.png", format="png") 
plt.show()



