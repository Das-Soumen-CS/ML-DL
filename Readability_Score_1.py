import textstat
import spacy
import colorama
from colorama import Fore, Back, Style

test_data = (
    "Dr. Bibhash Sen ,Dept. of Computer Science & Engineeirng"
    "PhD. in Engineering (Computer Science and Engineering"
    "Indian Institute of Engineering Science and Technology (IIEST), Shibpur, India."
    "M.E., Computer Science & Engineering"
    "Bengal Engineering and Science University (BESU), Shibpur, India."
    "B.Tech., Computer Science & Engineering"
    "North Eastern Regional Institute of Science and Technology (NERIST), India."

)


print(Fore.GREEN + "\n\n Given Text [ ] =\n",test_data,"\n")

#Returns the number of syllables present in the given text.
print(Fore.WHITE +"\n No of Syllable= ",textstat.syllable_count(test_data))

#Calculates the number of words present in the text. Optional removepunct specifies whether we need to
#take punctuation symbols into account while counting lexicons. Default value is True, 
#which removes the punctuation before counting lexicon items.
print("\n No of words =",textstat.lexicon_count(test_data, removepunct=True))

#Returns the number of sentences present in the given text.
print("\n No of Sentences =",textstat.sentence_count(test_data))

#Returns the number of characters present in the given text.
print("\n No of charercter =",textstat.char_count(test_data, ignore_spaces=True))

#Returns the number of characters present in the given text without punctuation.
print("\n No of charecter without punctuation =",textstat.letter_count(test_data, ignore_spaces=True))

#Returns the number of words with a syllable count equal to one.
print(" \n No of words with syllable count equal to 1 =",textstat.monosyllabcount(test_data))

#Returns the number of words with a syllable count greater than or equal to 3.
print("\n NO of words with a syllable count >=3 is =",textstat.polysyllabcount(test_data))
print(Fore.YELLOW + " \n flesh_reading_ease_score ==",textstat.flesch_reading_ease(test_data),"\n")
print(Fore.GREEN +"\n flesch_kincaid_grade_Score ==",textstat.flesch_kincaid_grade(test_data),"\n")
print(Fore.YELLOW +"\n smog_index_Score ==",textstat.smog_index(test_data),"\n")
print(Fore.GREEN +"\n coleman_liau_index_score=",textstat.coleman_liau_index(test_data))
print(Fore.YELLOW +"\n automated_readability_index_Score=",textstat.automated_readability_index(test_data))
print(Fore.GREEN +"\n dale_chall_readability_score=",textstat.dale_chall_readability_score(test_data))
print(Fore.YELLOW +"\n difficult_words_score=",textstat.difficult_words(test_data))
print(Fore.GREEN +"\n linsear_write_formula_Score= ",textstat.linsear_write_formula(test_data))
print(Fore.YELLOW +"\n gunning_fog_score=",textstat.gunning_fog(test_data))
print(Fore.GREEN +"\n text_standard=",textstat.text_standard(test_data))
print(Fore.YELLOW +"\n fernandez_huerta_Score ",textstat.fernandez_huerta(test_data))
print(Fore.GREEN +"\n szigriszt_pazos_Score=",textstat.szigriszt_pazos(test_data))
print(Fore.YELLOW +"\n gutierrez_polini_Score",textstat.gutierrez_polini(test_data))
print(Fore.GREEN +"\n crawford_Score",textstat.crawford(test_data))
print(Fore.YELLOW +"\n gulpease_index_Score=",textstat.gulpease_index(test_data))
print(Fore.GREEN +"\n osman_Score=",textstat.osman(test_data),"\n")
print(Fore.WHITE + "")

