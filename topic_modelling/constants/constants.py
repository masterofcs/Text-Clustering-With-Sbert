import os
import pathlib

root_path = str(pathlib.Path().absolute())


EMPTY_CACHE_TIME = os.environ.get('EMPTY_CACHE_TIME', 120) # seconds
torch_empty_cache_time = EMPTY_CACHE_TIME 


emojis_path = root_path + '/vn_support/emojicon.txt'
teencodes_path = root_path + '/vn_support/teencode.txt'
convert_to_vietnamese_path = root_path + '/vn_support/english-vnmese.txt'
wrong_words_path = root_path + '/vn_support/wrong-word.txt'
stopwords_path = root_path + '/vn_support/vietnamese-stopwords.txt'
map_languages = root_path + '/vn_support/iso_639_1_codes_langs.json'


lst_special_word = [   'không_thực_sự', 'không_thấy','không', 'không thể','không_những', 'chả', 
                'chẳng_hề', 'chẳng' 'không_có', 'không_có_gì_là', 'đâu_có', 'đâu', 'không_là'  
                'không_có_gì', 'không_có']

without_accent = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ'



lst_pos_tag = ['A','Ab','N', 'Np', 'Ny', 'V', 'Vb' 'Vy']


lst_dept_label_remove = ['coord', 'det', 'prp', 'punct']

#  depLabel : [pos_tag] 
lst_map_postag_label_remove = {'amod': ['R', 'P', 'A'], 'lov': ['E', 'A']}

lst_upper = ['PER', 'LOC', 'ORG', 'MISC']


emoji_dict = {} 
teen_dict = {} 
wrong_lst = [] 
english_dict = {}
lst_remove_words = [' i ', ' c ', ' _'] 
stopwords_lst = []

model_vncorenlp_path = root_path + '/VnCoreNLP/weights'

MODEL_VNCORENLP_MAX_HEAP_SIZE = os.environ.get('MODEL_VNCORENLP_MAX_HEAP_SIZE', '-Xmx10g')

max_heap_size = MODEL_VNCORENLP_MAX_HEAP_SIZE


sbert_model = 'sbert'



# sbert

MODEL_BATCH_SIZE = os.environ.get('MODEL_BATCH_SIZE', 32)
batch_size = MODEL_BATCH_SIZE

MODEL_SHOW_PROCESS = os.environ.get('MODEL_SHOW_PROCESS', False)
show_process = MODEL_SHOW_PROCESS

model_sbert_path = root_path + '/sbert/'
convert_to_tensor = True
convert_to_numpy = False
normalize_embeddings = True



### milvus
vector_dimension = 768

DOCUMENT_SIMILARITY_COLLECTION = os.environ.get('DOCUMENT_SIMILARITY_COLLECTION', 'DocumentSimilarity')

document_similarity_collection = DOCUMENT_SIMILARITY_COLLECTION