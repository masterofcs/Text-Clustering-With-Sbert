# import pandas as pd
# import numpy as np
import datetime
import re
import regex

import underthesea
import os
import langdetect

import constants.constants as default
import unicodedata
import os
import torch
import logging



def remove_urls(text):
    text = re.sub(r'(https|http)?:[^\s]+', ' ', text, flags=re.MULTILINE)
    return text

def remove_emojis(text):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                    "]+", re.UNICODE)
    return re.sub(emoj, '', text)

def normalize_unicode(text):
    return unicodedata.normalize('NFC', str(text))


def is_vietnamese_without_accent(text, list_without_accent):
    flag = False
    for i in list_without_accent:
        if i in text:
            flag = True
            break
    return flag



def is_vietnamese(text):
    try:
        return True if langdetect.detect(text) == 'vi' else False
    except:
        if len(text) > 50:
            print('error at: ', text[:20] + ' ... ' + text[-15:] + '\n \t ==> pass')
        else:
            print('error at: ', text + '\n \t ==> pass')
        return False


def is_language(text):
    try:
        return langdetect.detect(text)    
    except Exception as e:
        # print(e)
        # if len(text) > 50:
        #     print('error at: ', text[:20] + ' ... ' + text[-15:])
        # else:
        #     print('error at: ', text)
        return False

def is_vietnamese_without_accent(text, list_without_accent):
    flag = False
    for i in list_without_accent:
        if i in text:
            flag = True
            break

    return flag



class ProcessText:

    def __init__(self, text=''):
        self.text = text
    def setModel(self, model=None):
        self.model = model
    def set_text(self, text):
        self.text = text


    def get_text(self):
        return self.text

    @staticmethod
    def clean_text(  text, teen_dict=default.teen_dict, emoji_dict=default.emoji_dict,
                     english_dict=default.english_dict, wrong_lst=default.wrong_lst, 
                     punctuation_number = False):
        # document = str(document).lower()

        document = remove_urls(text)
        if len(emoji_dict) == 0:
            document = remove_emojis(document)

        # document = regex.sub(r'\.+', ".", document)
        
        # pattern = '[!,*)@#%(&$_?.^~:;’"]'
        # document = re.sub(pattern, '', document)

        new_sentence = ''
        for sentence in underthesea.sent_tokenize(document):
            # if not(sentence.isascii()):
            ###### CONVERT EMOJICON
            if len(emoji_dict) != 0:
                sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))

            ###### CONVERT TEENCODE
            if len(teen_dict) != 0:
                sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())

            ###### DEL Punctuation & Numbers
            if punctuation_number is not False:
                pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
                sentence = ' '.join(regex.findall(pattern,sentence))

            ####### Eng -> vietnamese
            if len(english_dict) != 0:
                sentence = ' '.join(english_dict[word] if word in english_dict else word for word in sentence.split())

            ###### DEL wrong words
            if len(wrong_lst) != 0:
                sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
            
            # new_sentence = new_sentence + sentence + '. '
            new_sentence = new_sentence + sentence + ' '


        document = new_sentence  
        #print(document)
        ###### DEL excess blank space
        document = regex.sub(r'\s+', ' ', document).strip()
        return document


    @staticmethod
    def process_special_word(text, list_word):
        new_text = ''
        text_lst = text.split(' ')
        i = 0

        flag = False
        for special_word in list_word:
            if special_word in text_lst:
                flag = True
        
        if flag == True:
            while i <= len(text_lst) - 1:
                word = text_lst[i]
                if  word in list_word:
                    next_idx = i+1
                    if next_idx <= len(text_lst) - 1:
                        word = word +'_'+ text_lst[next_idx]
                    i= next_idx + 1
                else:
                    i = i+1
                new_text = new_text + word + ' '


        else:
            new_text = text
        return new_text.strip()



    # tokenize vietnamese text
    @staticmethod
    def process_postag_vncorenlp(text, model, 
                                list_pos_tag=default.lst_pos_tag, 
                                list_dept_label_remove=default.lst_dept_label_remove, 
                                list_map_postag_label_remove=default.lst_map_postag_label_remove, 
                                list_upper=default.lst_upper):
        document_result = ''

        # for sentence in underthesea.sent_tokenize(text):  
        #     sentence = sentence.replace('.','')
        try:
            list_dict_words_postag = model.annotate_text(text)
        except Exception as e:
            print(e)
            return False

        if len(list_dict_words_postag) == 0:
            return False

        lst_key = list(list_map_postag_label_remove.keys())

        for index in list_dict_words_postag:
            new_document = ''

            for data in list_dict_words_postag[index]:
                pos_tag = data['posTag']
                dependency_label = data['depLabel'] 
                word = data['wordForm']
                ner_label = data['nerLabel']


                flag_upper = False
                for upper_key in list_upper:
                    if upper_key in ner_label:
                        flag_upper = True
                        break
                if flag_upper == False:
                    word = word.lower()
                            
                if (pos_tag in list_pos_tag) and (dependency_label not in list_dept_label_remove):
                    new_document = new_document + word + ' '

                for key in lst_key:
                    if dependency_label == key and pos_tag in list_map_postag_label_remove[key]:
                        new_document = new_document.replace(word, '')
                
            
            new_document = new_document + '. '
            # new_document = new_document.replace(new_document[0], new_document[0].upper())
            new_document = regex.sub(r'\s+', ' ', new_document).strip()
            document_result = document_result+new_document

        ###### DEL excess blank space
        # new_document = regex.sub(r'\s+', ' ', new_document).strip()
        # new_document = underthesea.word_tokenize(new_document, format="text")

        return document_result

    @staticmethod
    def remove_stopword(text, stopwords=default.stopwords_lst):
        ###### REMOVE stop words
        document = ' '.join('' if word in stopwords else word for word in text.split())

        #print(document)
        ###### DEL excess blank space
        # document = regex.sub(r'\s+', ' ', document).strip()
        return document

    @staticmethod
    def remove_redundant_words(text, list_remove_words):
        for word in list_remove_words:
            text = text.replace(word, ' ')


        ###### DEL excess blank space
        text = regex.sub(r'\s+', ' ', text).strip()
        return text

    def process_text_vncorenlp_text(self, s_text, nomalize = False,
                               teen_dict=default.teen_dict, emoji_dict=default.emoji_dict,
                               english_dict=default.english_dict, wrong_lst=default.wrong_lst,
                               punctuation_number = False,
                               list_remove_words = default.lst_remove_words,
                               stopwords=default.stopwords_lst):
        try:
            if nomalize:
                text = self.normalize_unicode(s_text)
            text = self.clean_text(text = s_text)
            text = underthesea.text_normalize(text)
            text = self.process_postag_vncorenlp(text = text, model=self.model)

            if text == False:
                return False

            text = self.process_special_word(text = text, list_word=default.lst_special_word)
            text = self.remove_stopword(text = text, stopwords=stopwords)
            text = self.remove_redundant_words(text = text, list_remove_words=list_remove_words)

            return text

        except Exception as e:
            if len(self.text) > 200:
                print('error at: ', self.text[:80] + ' ... ' + self.text[-30:])
            else:
                print('error at: ', self.text)

            logging.exception('message')


            # print(text)
            return False

    def process_text_vncorenlp(self, model, nomalize = False,
                                    teen_dict=default.teen_dict, emoji_dict=default.emoji_dict,
                                    english_dict=default.english_dict, wrong_lst=default.wrong_lst, 
                                    punctuation_number = False,
                                    list_remove_words = default.lst_remove_words, 
                                    stopwords=default.stopwords_lst):
        try:
            if nomalize:
                text = self.normalize_unicode(self.text)
            text = self.clean_text(text = self.text)
            text = underthesea.text_normalize(text)
            text = self.process_postag_vncorenlp(text = text, model=model)  

            if text == False:
                return False

            text = self.process_special_word(text = text, list_word=default.lst_special_word)
            text = self.remove_stopword(text = text, stopwords=stopwords)
            text = self.remove_redundant_words(text = text, list_remove_words=list_remove_words)

            return text   

        except Exception as e:
            if len(self.text) > 200:
                print('error at: ', self.text[:80] + ' ... ' + self.text[-30:])
            else:
                print('error at: ', self.text)

            logging.exception('message')


            # print(text)
            return False


    @staticmethod
    def is_vietnamese(text):
        try:
            return True if langdetect.detect(text) == 'vi' else False
        except:
            log = text[:20] + '...' + text[-15:] + '\n \t ==> pass'
            print(log)
            return False

    @staticmethod
    def is_language(text):
        return langdetect.detect(text)            

    @staticmethod
    def normalize_unicode(text):
        return unicodedata.normalize('NFC', str(text))



class SentenceTransformer_Process:

    def __init__(self, device = 'cpu',  sentences='', model=None):
        self.sentences = sentences
        self.device = device
        self.model = model

    def sentence_transformers_embedding(self, model,
                            batch_size = default.batch_size, 
                            show_progress_bar = default.show_process,
                            convert_to_tensor = default.convert_to_tensor,
                            normalize_embeddings = default.normalize_embeddings, 
                            convert_to_numpy=default.convert_to_numpy):

        data_embeddings = model.encode( sentences = self.sentences, 
                                        batch_size = batch_size,
                                        show_progress_bar = show_progress_bar,
                                        device = self.device,
                                        convert_to_tensor = convert_to_tensor,
                                        normalize_embeddings = normalize_embeddings)

        return data_embeddings

    def sentence_transformers_embedding_text(self, text,
                                             batch_size = default.batch_size,
                                             show_progress_bar = default.show_process,
                                             convert_to_tensor = default.convert_to_tensor,
                                             normalize_embeddings = default.normalize_embeddings,
                                             convert_to_numpy=default.convert_to_numpy):

        data_embeddings = self.model.encode( sentences = text,
                                             batch_size = batch_size,
                                             show_progress_bar = show_progress_bar,
                                             device = self.device,
                                             convert_to_tensor = convert_to_tensor,
                                             normalize_embeddings = normalize_embeddings)

        return data_embeddings
    
    def mean_pooling(model, attention_mask):
        token_embeddings = model[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def transformers_embedding(self, model, tokenizer):   
        encoding = tokenizer(   text=self.sentences,
                                add_special_tokens=True, # add '[CLS]' and '[SEP]
                                max_length=256,
                                # padding='max_length',  #static padding
                                padding='longest', #dynamic padding
                                truncation=True,
                                return_tensors='pt' #pytorch tensor
                            ).to(self.device)

        # # Compute token embeddings
        with torch.no_grad():
            output = model(**encoding, output_hidden_states=True, return_dict=True)

        embeddings = self.mean_pooling(output, encoding['attention_mask'])
        return embeddings