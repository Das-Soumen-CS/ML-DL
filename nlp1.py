def Read_File(arg):
	file=open(arg)
	file=file.read()
	print("The file type is: ",type(file))
	print("\n")
	print(file)
	print("\n")
	print("The no of charecters here is :",len(file))
	return file

def Sentance_Tokenizer(file):
	#print(file)
	from nltk import sent_tokenize
	sentences=sent_tokenize(file)
	print("\n")
	print("The total no of sentences here is :",len(sentences))
	print("\n")
	print(sentences)

def Word_ToKenizer(file):
	from nltk import word_tokenize
	word=word_tokenize(file)
	print("\n")
	print("The total no of  words here is :",len(word))
	print("\n")
	print("After Tokenizing th file we got :",word)
	return word

def Frequency_Calculate(file):
	from nltk.probability import FreqDist
	fdist=FreqDist(file)
	print("\n")
	print("Frequency of most_common 100 words listed here is as:",fdist.most_common(100))
	print("\n")
	import matplotlib.pyplot as plt 
	print("See the ploating of most common 10 words : ",fdist.plot(10))
	print("\n'")
	return fdist

def Remove_Punctuation(file):
	#print(file)
	words_without_punctuation=[]
	for i in file:
		if i.isalpha():
			words_without_punctuation.append(i.lower())
	print(words_without_punctuation)
	print("\n")
	print("The length of file words_without_punctuation",len(words_without_punctuation))
	print("\n")
	Frequency_Calculate(words_without_punctuation)
	return words_without_punctuation

def Remove_Stopwords(words):
	#print("hello",words)
	from nltk.corpus import stopwords
	stop_words=stopwords.words("english")
	#print(stop_words)
	words_without_stopwords=[]
	for j in words:
		if j not in stop_words:
			words_without_stopwords.append(j)
	print("Listed words without stop words :",words_without_stopwords)
	print("\n")
	print("After removing stop word length of file here is :",len(words_without_stopwords))
	Frequency_Calculate(words_without_stopwords)
	return words_without_stopwords

def Word_Cloud(file_path):
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from wordcloud import WordCloud,STOPWORDS
	from PIL import Image
	#char_mask=np.array(Image.open('Circle.png'))
	df=pd.read_csv(file_path)
	print(df.head())
	text = df.iloc[1].values
	wordcloud = WordCloud(width = 300,height = 200,background_color = 'black',stopwords = STOPWORDS).generate(str(text))
	#wordcloud=WordCloud(width = 300,height = 200,background_color = 'black',stopwords = STOPWORDS,mask=char_mask).generate(str(text))
	fig = plt.figure(figsize = (40, 30),facecolor = 'k',edgecolor = 'k')
	plt.imshow(wordcloud, interpolation = 'bilinear')
	plt.axis('off')
	plt.tight_layout(pad=0)
	plt.show()





def main():
	import sys
	file_path=sys.argv[1]
	import nltk
	file=Read_File(file_path)
	Sentance_Tokenizer(file)
	words=Word_ToKenizer(file)
	#print("Hi",words)
	Frequency_Calculate(file)
	words_without_punctuation=Remove_Punctuation(words)
	words_without_stopwords=Remove_Stopwords(words)
	cloud_fig=Word_Cloud(file_path)

main()